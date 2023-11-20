/*!
 * \file model_metadata.h
 * \brief Metadata stored in model lib
 */
#ifndef MLC_LLM_CPP_MODEL_METADATA_H_
#define MLC_LLM_CPP_MODEL_METADATA_H_
#define __STDC_FORMAT_MACROS
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>

#include <unordered_map>

namespace picojson {
class value;
using object = std::unordered_map<std::string, value>;
}  // namespace picojson

namespace mlc {
namespace llm {

struct ModelMetadata {
  struct Param {
    struct Preproc {
      tvm::runtime::String func_name;
      tvm::runtime::ShapeTuple out_shape;
      tvm::runtime::DataType out_dtype;
    };

    tvm::runtime::String name;
    tvm::runtime::ShapeTuple shape;
    tvm::runtime::DataType dtype;
    Preproc preproc;

    static Param FromJSON(const picojson::object& param_obj);
  };
  std::string model_type;
  std::string quantization;
  std::vector<Param> params;

  static ModelMetadata FromJSON(const picojson::object& json_str);
  static ModelMetadata FromModule(tvm::runtime::Module module);
};

}  // namespace llm
}  // namespace mlc

#endif