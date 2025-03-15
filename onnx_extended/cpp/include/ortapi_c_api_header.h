#pragma once

#include <onnxruntime_c_api.h>

#define IS_EMPTY(x) IS_EMPTY_HELPER(x)
#define IS_EMPTY_HELPER(x) IS_EMPTY_CHECK(x ## 1, 1)
#define IS_EMPTY_CHECK(a, b, ...) b

#if IS_EMPTY(ORT_EXPORT)
#define ORT_EXPORT __attribute__ ((visibility("default")))
#endif
