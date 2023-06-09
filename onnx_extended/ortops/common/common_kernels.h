#pragma once

#include "helpers.h"
#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

inline void _ThrowOnError_(OrtStatus* ort_status, const char* filename, int line, const OrtApi& api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(
          orthelpers::MakeString("error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
      }
    }
  }
}

#define ThrowOnError(api, ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

inline std::string KernelInfoGetOptionalAttributeString(const OrtApi& api, const OrtKernelInfo* info, const char* name, const std::string& default_value) {
  size_t size = 0;
  std::string out;

  OrtStatus* status = api.KernelInfoGetAttribute_string(info, name, nullptr, &size);

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
  out.resize(size);
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, name, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'
  return out;
}

inline int64_t KernelInfoGetOptionalAttributeInt64(const OrtApi& api, const OrtKernelInfo* info, const char* name, int64_t default_value) {
  int64_t out;
  OrtStatus* status = api.KernelInfoGetAttribute_int64(info, name, &out);

  if (status == nullptr) {
    return out;
  }
  OrtErrorCode code = api.GetErrorCode(status);
  if (code == ORT_FAIL) {
    api.ReleaseStatus(status);
    return default_value;
  }

  ThrowOnError(api, status);
  return default_value;
}

inline bool KernelInfoGetOptionalAttributeInt64AsBool(const OrtApi& api, const OrtKernelInfo* info, const char* name, bool default_value) {
  int64_t value = KernelInfoGetOptionalAttributeInt64(api, info, name, default_value ? 1 : 0);
  return value == 1;
}

} // namespace ortops
