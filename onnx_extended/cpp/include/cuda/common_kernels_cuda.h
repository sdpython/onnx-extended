#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include "onnx_extended_helpers.h"
#include <cuda_runtime.h>

namespace ortops {

inline const char *is_aligned(const void *ptr, std::size_t byte_count = 16) {
  if (ptr == nullptr)
    return "N";
  return (uintptr_t)ptr % byte_count == 0 ? "A" : "-";
}

inline const char *cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "<unknown>";
  }
}

inline const char *CudaDataTypeToString(cudaDataType_t dt) {
  // https://docs.nvidia.com/cuda/cuquantum/cutensornet/api/types.html
  switch (dt) {
  case CUDA_R_16F:
    return "CUDA_R_16F-2";
  case CUDA_R_16BF:
    return "CUDA_R_16BF-14";
  case CUDA_R_32F:
    return "CUDA_R_32F-0";
  case CUDA_R_64F:
    return "CUDA_R_64F-1";
  case CUDA_R_4I:
    return "CUDA_R_4I-16";
  case CUDA_R_8I:
    return "CUDA_R_8I-3";
  case CUDA_R_16I:
    return "CUDA_R_16I-20";
  case CUDA_R_32I:
    return "CUDA_R_32I-10";
  case CUDA_R_64I:
    return "CUDA_R_64I-24";
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
    return "CUDA_R_8F_E4M3-28";
  case CUDA_R_8F_E5M2:
    return "CUDA_R_8F_E5M2-29";
#endif
  default:
    return "<unknown>";
  }
}

inline const char *CublasComputeTypeToString(cublasComputeType_t ct) {
  // https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublas_api.h
  switch (ct) {
  case CUBLAS_COMPUTE_16F:
    return "CUBLAS_COMPUTE_16F-64";
  case CUBLAS_COMPUTE_32F:
    return "CUBLAS_COMPUTE_32F-68";
  case CUBLAS_COMPUTE_32I:
    return "CUBLAS_COMPUTE_32I-70";
  case CUBLAS_COMPUTE_32F_FAST_16F:
    return "CUBLAS_COMPUTE_32F_FAST_16F-74";
  case CUBLAS_COMPUTE_32F_FAST_16BF:
    return "CUBLAS_COMPUTE_32F_FAST_16BF-75";
  case CUBLAS_COMPUTE_32F_FAST_TF32:
    return "CUBLAS_COMPUTE_32F_FAST_TF32-77";
  case CUBLAS_COMPUTE_64F:
    return "CUBLAS_COMPUTE_64F-70";
  default:
    return "<unknown>";
  }
}

// It must exist somewhere already.
inline cudaDataType_t ToCudaDataType(ONNXTensorElementDataType element_type) {
  switch (element_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return CUDA_R_32F;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return CUDA_R_16F;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return CUDA_R_16BF;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && ORT_VERSION >= 1160
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    return CUDA_R_8F_E4M3;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    return CUDA_R_8F_E5M2;
#endif
  default:
#if defined(CUDA_VERSION)
    EXT_THROW("(ToCudaDataType) Unexpected element_type=", (int64_t)element_type,
              " CUDA_VERSION=", CUDA_VERSION, " ORT_VERSION=", ORT_VERSION, ".");
#else
    EXT_THROW("(ToCudaDataType) Unexpected element_type=", (int64_t)element_type,
              " (no CUDA), ORT_VERSION=", ORT_VERSION, ".");
#endif
  }
}

// It must exist somewhere already.
inline int32_t TypeSize(ONNXTensorElementDataType element_type) {
  switch (element_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && ORT_VERSION >= 1160
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    return 1;
#endif
  default:
#if defined(CUDA_VERSION)
    EXT_THROW("(TypeSize) Unexpected element_type=", element_type,
              " CUDA_VERSION=", CUDA_VERSION, " ORT_VERSION=", ORT_VERSION, ".");
#else
    EXT_THROW("(TypeSize) Unexpected element_type=", (int64_t)element_type,
              " (no CUDA), ORT_VERSION=", ORT_VERSION, ".");
#endif
  }
}

inline void _CublasThrowOnError_(cublasStatus_t status, const char *file, int line,
                                 const char *expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    EXT_THROW("Status=", cublasGetErrorEnum(status), " Expression [", expr, "] failed.\nFile ",
              file, ":", line);
  }
}

#define CUBLAS_THROW_IF_ERROR(expr) _CublasThrowOnError_((expr), __FILE__, __LINE__, #expr)

template <typename T>
void _check_cuda(T err, const char *const func, const char *const file, const int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(onnx_extended_helpers::MakeString(
        "CUDA error at: ", file, ":", line, "\n", cudaGetErrorString(err), " ", func, "\n"));
  }
}

#define CUDA_THROW_IF_ERROR(expr) _check_cuda((expr), #expr, __FILE__, __LINE__)

} // namespace ortops
