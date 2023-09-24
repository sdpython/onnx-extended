#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "onnx_extended_helpers.h"

namespace ortops {

////////////////////////
// errors and exceptions
////////////////////////

inline void _ThrowOnError_(OrtStatus *ort_status, const char *filename,
                           int line, const OrtApi &api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(onnx_extended_helpers::MakeString(
            "error: onnxruntime(", code, "), ", message, "\n    ", filename,
            ":", line));
      }
    }
  }
}

#define ThrowOnError(api, ort_status)                                          \
  _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

////////
// types
////////

inline bool is_float8(ONNXTensorElementDataType type) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && ORT_VERSION >= 1160
  return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ);
#else
  return type >= 17;
#endif
}

////////////////
// kernel inputs
////////////////

inline std::string KernelInfoGetInputName(const OrtApi &api,
                                          const OrtKernelInfo *info,
                                          int index) {
  std::size_t size;
  OrtStatus *status = api.KernelInfo_GetInputName(info, index, nullptr, &size);
  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return std::string();
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  std::string str_out;
  str_out.resize(size);
  ThrowOnError(api, api.KernelInfo_GetInputName(info, index, &str_out[0], &size));
  str_out.resize(size - 1); // remove the terminating character '\0'
  return str_out;
}

////////////////////
// kernel attributes
////////////////////

inline std::string KernelInfoGetOptionalAttributeString(
    const OrtApi &api, const OrtKernelInfo *info, const char *name,
    const std::string &default_value) {
  std::size_t size = 0;
  std::string str_out;

  OrtStatus *status =
      api.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return default_value;
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  str_out.resize(size);
  ThrowOnError(api,
               api.KernelInfoGetAttribute_string(info, name, &str_out[0], &size));
  str_out.resize(size - 1); // remove the terminating character '\0'
  return str_out;
}

template <typename T>
inline OrtStatus *KernelInfoGetAttributeApi(const OrtApi &api,
                                            const OrtKernelInfo *info,
                                            const char *name, T &out);

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<int64_t>(const OrtApi &api, const OrtKernelInfo *info,
                                   const char *name, int64_t &out) {
  return api.KernelInfoGetAttribute_int64(info, name, &out);
}

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<float>(const OrtApi &api, const OrtKernelInfo *info,
                                 const char *name, float &out) {
  return api.KernelInfoGetAttribute_float(info, name, &out);
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<std::vector<float>>(
    const OrtApi &api, const OrtKernelInfo *info, const char *name,
    std::vector<float> &out) {
  std::size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus *status =
      api.KernelInfoGetAttributeArray_float(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    status =
        api.KernelInfoGetAttributeArray_float(info, name, out.data(), &size);
  }

  return status;
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<std::vector<int64_t>>(
    const OrtApi &api, const OrtKernelInfo *info, const char *name,
    std::vector<int64_t> &out) {
  std::size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus *status =
      api.KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api, api.KernelInfoGetAttributeArray_int64(info, name,
                                                            out.data(), &size));
  }
  return status;
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<std::vector<std::string>>(
    const OrtApi &api, const OrtKernelInfo *info, const char *name,
    std::vector<std::string> &output) {
  EXT_THROW("Unable to retrieve attribute as an array of strings. "
            "You should use a single comma separated string.");
}

template <typename T>
inline T KernelInfoGetOptionalAttribute(const OrtApi &api,
                                        const OrtKernelInfo *info,
                                        const char *name, T default_value) {
  T out;
  OrtStatus *status = KernelInfoGetAttributeApi<T>(api, info, name, out);

  if (status == nullptr)
    return out;
  OrtErrorCode code = api.GetErrorCode(status);
  if (code == ORT_FAIL) {
    api.ReleaseStatus(status);
    return default_value;
  }

  ThrowOnError(api, status);
  return default_value;
}

inline bool KernelInfoGetOptionalAttributeInt64AsBool(const OrtApi &api,
                                                      const OrtKernelInfo *info,
                                                      const char *name,
                                                      bool default_value) {
  int64_t value = KernelInfoGetOptionalAttribute<int64_t>(
      api, info, name, default_value ? 1 : 0);
  return value == 1;
}

} // namespace ortops
