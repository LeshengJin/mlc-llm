#include "ndarray_cache_metadata.h"

namespace mlc {
namespace llm {
namespace loader {

using namespace tvm::runtime;
using Device = tvm::Device;

template <typename ExpectedType>
ExpectedType AsType(const picojson::value& json) {
  ICHECK(json.is<ExpectedType>());
  return json.get<ExpectedType>();
}

template <typename ValueType>
ValueType GetValue(const picojson::object& json, const std::string& key) {
  return AsType<ValueType>(json.at(key));
}

NDArrayCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const picojson::object& json) {
  std::vector<ShapeTuple::index_type> shape;
  {
    picojson::array shape_json = GetValue<picojson::array>(json, "shape");
    shape.reserve(shape_json.size());
    for (const picojson::value& d : shape_json) {
      shape.push_back(AsType<int64_t>(d));
    }
  }
  NDArrayCacheMetadata::FileRecord::ParamRecord result;
  std::string dtype = GetValue<std::string>(json, "dtype");
  result.name = GetValue<std::string>(json, "name");
  result.dtype = DataType(String2DLDataType(dtype));
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.byte_offset = GetValue<int64_t>(json, "byteOffset");
  result.shape = ShapeTuple(std::move(shape));
  return result;
}

NDArrayCacheMetadata::FileRecord JSONAsFileRecord(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  NDArrayCacheMetadata::FileRecord result;
  result.data_path = GetValue<std::string>(json, "dataPath");
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsParamRecord(AsType<picojson::object>(item)));
  }
  return result;
}

NDArrayCacheMetadata JSONAsNDArrayCacheMetadata(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  NDArrayCacheMetadata result;
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsFileRecord(AsType<picojson::object>(item)));
  }
  return result;
}

NDArrayCacheMetadata NDArrayCacheMetadata::LoadFromStr(const std::string& json_str,
                                                       const std::string& path) {
  picojson::value json_info;
  picojson::parse(json_info, json_str);
  NDArrayCacheMetadata result = JSONAsNDArrayCacheMetadata(AsType<picojson::object>(json_info));
  result.path = path;
  return result;
}

ShardInfo::TensorInfo LoadTensorInfoFromJSON(const picojson::array& json_tensor_info) {
  CHECK_EQ(json_tensor_info.size(), 2) << "ValueError: Invalid tensor info JSON";
  picojson::array shape_json = AsType<picojson::array>(json_tensor_info[0]);
  int ndim = shape_json.size();
  std::vector<int64_t> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(AsType<int64_t>(shape_json[i]));
  }
  std::string dtype = AsType<std::string>(json_tensor_info[1]);
  return ShardInfo::TensorInfo{ShapeTuple(std::move(shape)), DataType(String2DLDataType(dtype))};
}

ShardInfo::ShardFunc LoadShardFuncFromJSON(const picojson::array& json_shard_func) {
  int n = json_shard_func.size();
  ShardInfo::ShardFunc shard_info;
  shard_info.name = AsType<std::string>(json_shard_func[0]);
  shard_info.output_info = LoadTensorInfoFromJSON(AsType<picojson::array>(json_shard_func[1]));
  shard_info.params.reserve(n - 2);
  for (int i = 2; i < n; ++i) {
    shard_info.params.push_back(AsType<int64_t>(json_shard_func[i]));
  }
  return shard_info;
}

std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str) {
  picojson::value json_info;
  picojson::parse(json_info, json_str);
  picojson::object json_obj = AsType<picojson::object>(json_info);
  std::unordered_map<std::string, ShardInfo> result;
  for (auto kv : json_obj) {
    std::string name = kv.first;
    picojson::array json_shard_funcs = AsType<picojson::array>(kv.second);
    ShardInfo info;
    std::vector<ShardInfo::ShardFunc>& shard_funcs = info.funcs;
    shard_funcs.reserve(json_shard_funcs.size());
    for (const picojson::value& json_shard_func : json_shard_funcs) {
      shard_funcs.push_back(LoadShardFuncFromJSON(AsType<picojson::array>(json_shard_func)));
    }
    result[name] = info;
  }
  return result;
}

NDArray NDArrayCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data,
    std::function<void(NDArray, const void*, int64_t)> f_load) const {
  NDArray arr = NDArray::Empty(shape, dtype, device);
  if (dtype == DataType::Float(32) && format == "f32-to-bf16") {
    // decode bf16 to f32
    std::vector<uint16_t> buffer(nbytes / 2);
    std::vector<uint32_t> decoded(nbytes / 2);
    std::memcpy(buffer.data(), raw_data->data() + byte_offset, nbytes);
    for (size_t i = 0; i < buffer.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    f_load(arr, decoded.data(), decoded.size() * sizeof(uint32_t));
  } else {
    f_load(arr, raw_data->data() + byte_offset, nbytes);
  }
  return arr;
}

}  // namespace loader
}  // namespace llm
}  // namespace mlc
