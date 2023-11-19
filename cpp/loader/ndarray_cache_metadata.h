#ifndef MLC_LLM_CPP_LOADER_NDARRAY_CACHE_METADATA_H_
#define MLC_LLM_CPP_LOADER_NDARRAY_CACHE_METADATA_H_

#define PICOJSON_USE_INT64

#include <picojson.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>

namespace mlc {
namespace llm {
namespace loader {

/*!
 * \brief Metadata for NDArray cache, which by default, is named as "ndarray-cache.json".
 */
struct NDArrayCacheMetadata {
  /*! \brief Each shard of NDArray cache, which by default, is named as "params_shard_x.bin". */
  struct FileRecord {
    /*! \brief Metadata of each parameter */
    struct ParamRecord {
      /*!
       * \brief Load the parameter from raw data.
       * \param device The device to load the parameter onto.
       * \param raw_data The raw data stream
       * \param f_load The function to load the parameter from raw data.
       */
      tvm::runtime::NDArray Load(
          tvm::Device device, const std::string* raw_data,
          std::function<void(tvm::runtime::NDArray, const void*, int64_t)> f_load) const;

      /*! \brief Name of the parameter */
      std::string name;
      /*! \brief Shape of the parameter */
      tvm::runtime::ShapeTuple shape;
      /*! \brief Data type of the parameter */
      tvm::runtime::DataType dtype;
      /*! \brief Format of the parameter */
      std::string format;
      /*! \brief Number of bytes */
      int64_t nbytes;
      /*! \brief Offset from the raw stream */
      int64_t byte_offset;
    };

    /*! \brief Relative path to the bin file */
    std::string data_path;
    /*! \brief Format of the file */
    std::string format;
    /*! \brief Size of the file */
    int64_t nbytes;
    /*! \brief The parameters in the file */
    std::vector<ParamRecord> records;
  };
  /*! \brief The files in the NDArray cache */
  std::vector<FileRecord> records;
  /*! \brief The path to the `ndarray-cache.json` file */
  std::string path;

  /*! \brief Load the metadata from a specific path */
  static NDArrayCacheMetadata LoadFromStr(const std::string& json_str, const std::string& path);
};

/*!
 * \brief Information of sharding function,
 * including the shard function name and extra parameters.
 */
struct ShardInfo {
  struct TensorInfo {
    tvm::runtime::ShapeTuple shape;
    tvm::runtime::DataType dtype;
  };
  struct ShardFunc {
    std::string name;
    TensorInfo output_info;
    std::vector<int64_t> params;
  };
  std::vector<ShardFunc> funcs;
};

/*!
 * \brief Load the shard information from dist
 * \param path Path to the file to be loaded
 * \return Mapping from parameter name to its shard dim
 */
std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str);

}  // namespace loader
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_LOADER_NDARRAY_CACHE_METADATA_H_