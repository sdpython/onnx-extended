#pragma once

#include "onnx_extended_helpers.h"
#include <stdexcept>

#define NVTE_ERROR(x)                                                          \
  do {                                                                         \
    throw std::runtime_error(onnx_extended_helpers::MakeString(                \
        __FILE__, ":", __LINE__, " in function ", __func__, ": ", x));         \
  } while (false)

#define NVTE_CHECK(x, ...)                                                     \
  do {                                                                         \
    if (!(x)) {                                                                \
      NVTE_ERROR(std::string("Assertion failed: " #x ". ") +                   \
                 std::string(__VA_ARGS__));                                    \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_CUDA(ans)                                                   \
  {                                                                            \
    auto status = ans;                                                         \
    NVTE_CHECK(status == cudaSuccess,                                          \
               "CUDA Error: " + std::string(cudaGetErrorString(status)));      \
  }

#define NVTE_CHECK_CUBLAS(ans)                                                 \
  {                                                                            \
    auto status = ans;                                                         \
    NVTE_CHECK(status == CUBLAS_STATUS_SUCCESS,                                \
               "CUBLAS Error: " + std::string(cublasGetStatusString(status))); \
  }

#define checkCudaErrors(val) _check_cuda((val), #val, __FILE__, __LINE__)

template <typename T>
void _check_cuda(T err, const char *const func, const char *const file,
                 const int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(onnx_extended_helpers::MakeString(
        "CUDA error at: ", file, ":", line, "\n", cudaGetErrorString(err), " ",
        func, "\n"));
  }
}
