#pragma once

#include "onnx_extended_helpers.h"
#include "ortapi_version.h"

namespace ortapi {

inline static const OrtApi *GetOrtApi() {
  const OrtApi *api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION_SUPPORTED);
  return api_;
}

inline const char *ort_version() { return OrtGetApiBase()->GetVersionString(); }

inline void _ThrowOnError_(OrtStatus *ort_status, const char *filename, int line) {
  if (ort_status) {
    std::string message(GetOrtApi()->GetErrorMessage(ort_status));
    OrtErrorCode code = GetOrtApi()->GetErrorCode(ort_status);
    throw std::runtime_error(onnx_extended_helpers::MakeString(
        "error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
  }
}

#define ThrowOnError(ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__)

} // namespace ortapi
