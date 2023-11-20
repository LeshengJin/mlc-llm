#ifndef MLC_LLM_CPP_LOADER_MULTI_GPU_LOADER_H_
#define MLC_LLM_CPP_LOADER_MULTI_GPU_LOADER_H_

#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/worker.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "ndarray_cache_metadata.h"

namespace mlc {
namespace llm {
namespace loader {

using namespace tvm::runtime;
using Device = tvm::Device;
using FileRecord = NDArrayCacheMetadata::FileRecord;
using ParamRecord = NDArrayCacheMetadata::FileRecord::ParamRecord;

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*! \brief Create a shard loader. */
  static ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                          std::string shard_info, Module mod, tvm::runtime::Module local_vm);
  /*! \brief Load the i-th parameter */
  NDArray Load(int weight_index) const;

  NDArray LoadParamOnWorker0(int weight_index) const;

  /*! \brief Load all the parameters */
  Array<NDArray> LoadAll() const;

  NDArray ApplyShardFunc(const ShardInfo::ShardFunc& shard_func, const NDArray& param) const;

  /*! \brief Load all the pre-sharded parameters */
  Array<NDArray> LoadAllPresharded() const;

  /*! \brief Load the i-th parameter from presharded binaries */
  NDArray LoadPresharded(int weight_index) const;

  /*! \brief Slice the given tensor at a specific dimension */
  NDArray Shard(NDArray source, int dim, int num_slices) const;

  static constexpr const char* _type_key = "runtime.disco.ShardLoader";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShardLoaderObj, Object);

 public:
  /*! \brief Information of how each weight is stored and sharded */
  struct ParamInfo {
    const FileRecord* file;
    const ParamRecord* param;
    ShardInfo shard_info;
  };
  /*! \brief The PackedFuncs being used during sharding */
  std::unordered_map<std::string, PackedFunc> shard_funcs_;
  /*! \brief The metadata loaded from `ndarray-cache.json` */
  NDArrayCacheMetadata metadata_;
  /*! \brief Sharding information for each weight */
  std::vector<ParamInfo> param_info_;
  /*! \brief Maps the name of a shard to its index */
  std::unordered_map<std::string, int> param_name_to_index_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;

 private:
  /*! \brief Load the i-th parameter without post-processing
   *
   * This function should not be called externally, as it does not
   * check for post-processing that may be required.  Instead, the
   * public function `Load` or `LoadPresharded` should be called.
   *
   * \param weight_index The index of NDArray tensor to load
   *
   * \returns The full tensor at the specified index
   */
  NDArray LoadDirect(int weight_index) const;
};

TVM_REGISTER_OBJECT_TYPE(ShardLoaderObj);

}  // namespace loader
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_LOADER_MULTI_GPU_LOADER_H_
