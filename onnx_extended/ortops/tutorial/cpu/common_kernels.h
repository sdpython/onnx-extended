#pragma once

#include "helpers.h"
#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

inline void _ThrowOnError_(OrtStatus* ort_status, const char* filename, int line, const OrtApi& api) {
    if (ort_status) {
        std::string message(api.GetErrorMessage(ort_status));
        OrtErrorCode code = api.GetErrorCode(ort_status);
        throw std::runtime_error(
            orthelpers::MakeString("error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
    }
}

#define ThrowOnError(api, ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

} // namespace ortops
