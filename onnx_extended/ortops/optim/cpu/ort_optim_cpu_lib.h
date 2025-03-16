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
