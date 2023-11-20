#define __STDC_FORMAT_MACROS
#include "./model_metadata.h"

#include <tvm/runtime/packed_func.h>

#include "./json_parser.h"

namespace mlc {
namespace llm {

template <typename ExpectedType>
ExpectedType AsType(const picojson::value& json) {
  ICHECK(json.is<ExpectedType>());
  return json.get<ExpectedType>();
}

using namespace tvm::runtime;

ModelMetadata::Param::Preproc ModelMetadata::Param::Preproc::FromJSON(
    const picojson::array& json_preproc) {
  Preproc preproc;
  CHECK_EQ(json_preproc.size(), 2) << "ValueError: Invalid preprocessing info in JSON";
  preproc.func_name = AsType<std::string>(json_preproc[0]);
  picojson::array shape_dtype = AsType<picojson::array>(json_preproc[1]);
  CHECK_EQ(json_preproc.size(), 2) << "ValueError: Invalid preprocessing dtype shape in JSON";
  picojson::array shape_json = AsType<picojson::array>(shape_dtype[0]);
  int ndim = shape_json.size();
  std::vector<int64_t> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(AsType<int64_t>(shape_json[i]));
  }
  std::string dtype = AsType<std::string>(shape_dtype[1]);
  preproc.out_shape = ShapeTuple(std::move(shape));
  preproc.out_dtype = DataType(String2DLDataType(dtype));
  return preproc;
}

ModelMetadata::Param ModelMetadata::Param::FromJSON(const picojson::object& param) {
  Param result;
  result.name = json::Lookup<std::string>(param, "name");
  result.shape = json::Lookup<ShapeTuple>(param, "shape");
  result.dtype = json::Lookup<DataType>(param, "dtype");
  picojson::array preprocs = json::Lookup<picojson::array>(param, "shard_info");
  result.preprocs.reserve(preprocs.size());
  for (const picojson::value& json_preproc : preprocs) {
    result.preprocs.emplace_back(
        ModelMetadata::Param::Preproc::FromJSON(AsType<picojson::array>(json_preproc)));
  }
  return result;
}

ModelMetadata ModelMetadata::FromJSON(const picojson::object& metadata) {
  ModelMetadata result;
  result.model_type = json::Lookup<std::string>(metadata, "model_type");
  result.quantization = json::Lookup<std::string>(metadata, "quantization");
  picojson::array params = json::Lookup<picojson::array>(metadata, "params");
  result.params.reserve(params.size());
  for (const picojson::value& json_param : params) {
    result.params.emplace_back(ModelMetadata::Param::FromJSON(json::AsJSONObject(json_param)));
  }
  return result;
}

ModelMetadata ModelMetadata::FromModule(tvm::runtime::Module module) {
  std::string json_str = "";
  try {
    TypedPackedFunc<String()> pf = module.GetFunction("_metadata");
    ICHECK(pf != nullptr);
    json_str = pf();
  } catch (...) {
    return ModelMetadata();  // TODO: add a warning message about legacy usecases
  }
  picojson::object json = json::ParseObject(json_str);
  try {
    return ModelMetadata::FromJSON(json);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str;
    throw e;
  }
}

std::string GetModelMetadataStringFromModule(tvm::runtime::Module module) {
  std::string json_str = "";
  try {
    TypedPackedFunc<String()> pf = module.GetFunction("_metadata");
    ICHECK(pf != nullptr);
    json_str = pf();
  } catch (...) {
    return "";  // TODO: add a warning message about legacy usecases
  }
  return json_str;
}

// std::string GetModelMetadataStringFromModule(PackedFunc get_metadata) {
//   ObjectRef ret = get_metadata();
//   std::string metadata = std::string(Downcast<String> ret);
//   return metadata;
// }

ModelMetadata ModelMetadata::FromString(std::string json_str) {
  picojson::object json = json::ParseObject(json_str);
  try {
    return ModelMetadata::FromJSON(json);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str;
    throw e;
  }
}

}  // namespace llm
}  // namespace mlc
