#include "multi_gpu_loader.h"

#include "../model_metadata.h"
namespace mlc {
namespace llm {
namespace loader {

void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

inline int64_t IntegerFromShapeTuple(const ShapeTuple& shape) {
  CHECK_EQ(shape.size(), 1) << "ValueError: shape tuple must be 1-d to be converted to integer.";
  return shape[0];
}

std::unordered_map<std::string, ShardInfo> GetShardInfoMap(ModelMetadata model_metadata) {
  std::unordered_map<std::string, ShardInfo> shards;
  for (ModelMetadata::Param param : model_metadata.params) {
    ShardInfo shard_info;
    ShardInfo::ShardFunc shard_func;
    ShardInfo::TensorInfo output_info;
    output_info.shape = param.preproc.out_shape;
    output_info.dtype = param.preproc.out_dtype;
    shard_func.name = param.preproc.func_name;
    shard_func.output_info = output_info;
    shard_info.funcs.emplace_back(shard_func);
    shards[param.name] = shard_info;
  }
  return shards;
}

ObjectRef ShardLoaderObj::Create(const std::string& path_to_metadata, const std::string& metadata,
                                 std::string shard_info, Module mod,
                                 tvm::runtime::Module local_vm) {
  if (shard_info.empty() && mod.defined()) {
    if (PackedFunc get_shard_info = mod->GetFunction("get_shard_info"); get_shard_info != nullptr) {
      shard_info = get_shard_info().operator String();
    }
  }
  ObjectPtr<ShardLoaderObj> n = make_object<ShardLoaderObj>();
  n->metadata_ = NDArrayCacheMetadata::LoadFromStr(metadata, path_to_metadata);
  n->current_file_ = nullptr;
  n->param_info_.clear();
  ModelMetadata model_metadata_ = ModelMetadata::FromModule(local_vm);
  std::unordered_map<std::string, ShardInfo> shards;
  if (model_metadata_.params.empty()) {
    shards = LoadShardInfoFromStr(shard_info);
  } else {
    shards = GetShardInfoMap(model_metadata_);
  }
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      const std::string& name = param_record.name;
      int index = n->param_info_.size();
      n->param_name_to_index_[name] = index;
      ShardInfo& shard_info = shards[name];
      for (const ShardInfo::ShardFunc& shard_func : shard_info.funcs) {
        const std::string& name = shard_func.name;
        if (PackedFunc f = mod.defined() ? mod->GetFunction(name, true) : nullptr; f != nullptr) {
          n->shard_funcs_[name] = f;
        } else if (const PackedFunc* f = tvm::runtime::Registry::Get(name)) {
          n->shard_funcs_[name] = *f;
        } else {
          LOG(FATAL) << "ValueError: Undefined function: " << name;
        }
      }
      n->param_info_.emplace_back(ParamInfo{&file_record, &param_record, shard_info});
    }
  }
  return ObjectRef(std::move(n));
}

NDArray ShardLoaderObj::ApplyShardFunc(const ShardInfo::ShardFunc& shard_func,
                                       const NDArray& param) const {
  Device device = param->device;
  NDArray o = NDArray::Empty(shard_func.output_info.shape, shard_func.output_info.dtype, device);
  PackedFunc f = this->shard_funcs_.at(shard_func.name);
  int n = static_cast<int>(shard_func.params.size());
  std::vector<TVMValue> tvm_args(n + 2);
  std::vector<int> type_codes(n + 2);
  TVMArgsSetter setter(tvm_args.data(), type_codes.data());
  const DLTensor* w_in = param.operator->();
  const DLTensor* w_out = o.operator->();
  setter(0, const_cast<DLTensor*>(w_in));
  for (int i = 0; i < n; ++i) {
    setter(i + 1, shard_func.params[i]);
  }
  setter(n + 1, const_cast<DLTensor*>(w_out));
  TVMRetValue rv;
  f.CallPacked(TVMArgs(tvm_args.data(), type_codes.data(), n + 2), &rv);
  return o;
}

std::string GetSiblingPath(const std::string& path, const std::string& filename) {
  size_t found = path.find_last_of("/\\");
  if (found != std::string::npos) {
    return path.substr(0, found + 1) + filename;
  }
  LOG(FATAL) << "ValueError: Cannot find the parent directory: " << path;
}

NDArray ShardLoaderObj::LoadParamOnWorker0(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  Device device = worker->default_device;
  int param_index = param_name_to_index_.at("param_" + std::to_string(weight_index));
  const ParamInfo& param_info = param_info_.at(param_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  auto load = [this, param, device, file]() {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      LoadBinaryFromFile(file_name, &this->current_file_stream_);
    }
    return param->Load(
        device, &this->current_file_stream_,
        [](NDArray param, const void* data, size_t nbytes) { param.CopyFromBytes(data, nbytes); });
  };

  if (worker_id == 0) {
    NDArray w = load();
    return w;
  } else {
    NDArray w = NDArray::Empty(param->shape, param->dtype, device);
    return w;
  }
}

std::tuple<int, int> ParseParamShardingInfo(const ParamRecord* param) {
  // Given a name "param_shard-X-of-Y", return the integer values
  // rank=(X-1) and world_size=Y.

  std::string name = param->name;
  size_t pos1 = name.rfind("-of-");
  CHECK(pos1 != std::string::npos)
      << "Attempt to read num_shards from unexpected param name: " << name;
  size_t pos2 = name.rfind("_shard-", pos1 - 1);
  CHECK(pos2 != std::string::npos)
      << "Attempt to read sharded worker_id from unexpected param name: " << name;

  int num_shards = std::stoi(name.substr(pos1 + 4));
  int worker_id = std::stoi(name.substr(pos2 + 7, pos1 - pos2 - 7)) - 1;

  CHECK_GT(num_shards, 1);
  CHECK_GE(worker_id, 0);
  CHECK_LT(worker_id, num_shards);

  return {num_shards, worker_id};
}

NDArray ShardLoaderObj::LoadDirect(int weight_index) const {
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;

  if (file != current_file_) {
    current_file_ = file;
    std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
    LoadBinaryFromFile(file_name, &this->current_file_stream_);
  }
  return param->Load(
      device, &this->current_file_stream_,
      [](NDArray param, const void* data, size_t nbytes) { param.CopyFromBytes(data, nbytes); });
}

NDArray ShardLoaderObj::Load(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  Device device = worker->default_device;
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;

  bool needs_sharding = !param_info.shard_info.funcs.empty();
  if (needs_sharding) {
    ShapeTuple shape = param_info.shard_info.funcs.back().output_info.shape;
    DataType dtype = param_info.shard_info.funcs.back().output_info.dtype;
    ICHECK(shape.size() >= 1 && shape[0] == num_shards)
        << "ValueError: The first dimension of the "
        << "output shape must be equal to the "
        << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
    NDArray recv = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
    if (worker_id == 0) {
      NDArray w = LoadDirect(weight_index);
      for (const ShardInfo::ShardFunc& shard_func : param_info.shard_info.funcs) {
        w = this->ApplyShardFunc(shard_func, w);
      }
      ScatterFromWorker0(w, recv);
    } else {
      ScatterFromWorker0(tvm::NullOpt, recv);
    }
    return recv;
  } else {
    if (worker_id == 0) {
      NDArray w = LoadDirect(weight_index);
      BroadcastFromWorker0(w, w);
      return w;
    } else {
      NDArray w = NDArray::Empty(param->shape, param->dtype, device);
      BroadcastFromWorker0(w, w);
      return w;
    }
  }
}

Array<NDArray> ShardLoaderObj::LoadAll() const {
  int n = static_cast<int>(param_info_.size());
  Array<NDArray> shards;
  shards.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::string param_name = "param_" + std::to_string(i);
    ICHECK(this->param_name_to_index_.count(param_name));
    int shard_id = this->param_name_to_index_.at(param_name);
    shards.push_back(this->Load(shard_id));
  }
  return shards;
}

NDArray ShardLoaderObj::LoadPresharded(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  size_t num_weights = param_info_.size() / num_shards;
  size_t index = worker_id * num_weights + weight_index;
  CHECK(index < param_info_.size())
      << "Loading param " << weight_index << " for shard " << worker_id << " at position " << index
      << " is out of bounds for the provided ndarray chace.";

  const auto& shard_info = param_info_[index];
  const ParamRecord* param = shard_info.param;
  const FileRecord* file = shard_info.file;

  auto [p_num_shards, p_worker_id] = ParseParamShardingInfo(param);
  CHECK_EQ(num_shards, p_num_shards)
      << "Runtime number of shards (" << num_shards
      << ") does not match number of compiled shards (" << p_num_shards << "): " << param->name
      << " loaded from " << file->data_path;
  CHECK_EQ(worker_id, p_worker_id)
      << "Runtime worker_id (" << worker_id << ") does not match worker_id of compiled shard ("
      << p_worker_id << "): " << param->name << " loaded from " << file->data_path;

  return LoadDirect(index);
}

Array<NDArray> ShardLoaderObj::LoadAllPresharded() const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  size_t worker_id = static_cast<size_t>(worker->worker_id);
  size_t num_workers = static_cast<size_t>(worker->num_workers);
  size_t num_params = param_info_.size() / num_workers;

  Array<NDArray> params;
  params.reserve(num_params);
  for (size_t i_param = 0; i_param < num_params; ++i_param) {
    std::string param_name = static_cast<const std::stringstream&>(
                                 std::stringstream() << "param_" << i_param << "_shard-"
                                                     << (worker_id + 1) << "-of-" << num_workers)
                                 .str();

    auto it = param_name_to_index_.find(param_name);
    CHECK(it != param_name_to_index_.end())
        << "Parameter " << param_name << " was not found in the parameter set";
    int param_id = this->param_name_to_index_.at(param_name);
    params.push_back(this->LoadDirect(param_id));
  }
  return params;
}

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoader").set_body_typed(ShardLoaderObj::Create);
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoad")
    .set_body_typed([](ObjectRef loader_obj, ShapeTuple weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->Load(IntegerFromShapeTuple(weight_index));
    });
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadPresharded")
    .set_body_typed([](ObjectRef loader_obj, ShapeTuple weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadPresharded(IntegerFromShapeTuple(weight_index));
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAll").set_body_typed([](ObjectRef loader_obj) {
  const auto* loader = loader_obj.as<ShardLoaderObj>();
  CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                           << loader_obj->GetTypeKey();
  return loader->LoadAll();
});

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAllPresharded")
    .set_body_typed([](ObjectRef loader_obj) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadAllPresharded();
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadParamOnWorker0")
    .set_body_typed([](ObjectRef loader_obj, int param_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadParamOnWorker0(param_index);
    });
}  // namespace loader
}  // namespace llm
}  // namespace mlc