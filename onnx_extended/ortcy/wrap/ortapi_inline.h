#pragma once

#include "helpers.h"
#include "onnxruntime_c_api.h"

namespace ortapi {

inline const OrtApi *GetOrtApi() {
  static const OrtApi *api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return api_;
}

const char* ort_version() {
    return "GetOrtApi()->GetBuildInfoString();";
}

inline void _ThrowOnError_(OrtStatus* ort_status, const char* filename, int line) {
    if (ort_status) {
        std::string message(GetOrtApi()->GetErrorMessage(ort_status));
        OrtErrorCode code = GetOrtApi()->GetErrorCode(ort_status);
        throw std::runtime_error(
            MakeString("error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
    }
}

#define ThrowOnError(ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__)

} // namespace ortapi
