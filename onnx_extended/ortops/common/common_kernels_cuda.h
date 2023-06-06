#pragma once

#include "cublas_v2.h"
#include "helpers.h"
#include <cuda_runtime.h>

namespace ortops {

static const char* is_aligned(const void *ptr, size_t byte_count=16) {
  if (ptr == nullptr) return "N";
  return (uintptr_t)ptr % byte_count == 0 ? "A" : "-";
}

static const char* cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "<unknown>";
  }
}

static const char* CudaDataTypeToString(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_16F: return "CUDA_R_16F";
    case CUDA_R_16BF: return "CUDA_R_16BF";
    case CUDA_R_32F: return "CUDA_R_32F";
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    case CUDA_R_8F_E4M3: return "CUDA_R_8F_E4M3";
    case CUDA_R_8F_E5M2: return "CUDA_R_8F_E5M2";
#endif
    default: return "<unknown>";
  }
}

static const char* CublasComputeTypeToString(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_16F: return "CUBLAS_COMPUTE_16F";
    case CUBLAS_COMPUTE_32F: return "CUBLAS_COMPUTE_32F";
    case CUBLAS_COMPUTE_32F_FAST_16F: return "CUBLAS_COMPUTE_32F_FAST_16F";
    case CUBLAS_COMPUTE_32F_FAST_16BF: return "CUBLAS_COMPUTE_32F_FAST_16BF";
    case CUBLAS_COMPUTE_32F_FAST_TF32: return "CUBLAS_COMPUTE_32F_FAST_TF32";
    case CUBLAS_COMPUTE_64F: return "CUBLAS_COMPUTE_64F";
    default: return "<unknown>";
  }
}

// It must exist somewhere already.
cudaDataType_t ToCudaDataType(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return CUDA_R_32F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return CUDA_R_16F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return CUDA_R_16BF;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return CUDA_R_8F_E4M3;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return CUDA_R_8F_E5M2;
#endif
    default:
#if defined(CUDA_VERSION)
      EXT_THROW("Unexpected element_type=", element_type, " CUDA_VERSION=", #CUDA_VERSION, ".");
#else
      EXT_THROW("Unexpected element_type=", element_type, " (no CUDA).");
#endif
  }
}

// It must exist somewhere already.
int32_t TypeSize(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return 1;
#endif
    default:
#if defined(CUDA_VERSION)
      EXT_THROW("Unexpected element_type=", element_type, " CUDA_VERSION=", CUDA_VERSION, ".");
#else
      EXT_THROW("Unexpected element_type=", element_type, " (no CUDA).");
#endif
  }
}

void _CublasThrowOnError_(cublasStatus_t status, const char* file, int line, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    EXT_THROW("Expression [", expr, "] failed.\nFile ", file, ":", line);
  }
}

#define CUBLAS_THROW_IF_ERROR(expr) _CublasThrowOnError_((expr), __FILE__, __LINE__, #expr)

} // namespace ortops
