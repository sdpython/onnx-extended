// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library
#pragma once

#include "ortapi_c_api_header.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api_base);

#ifdef __cplusplus
}
#endif
