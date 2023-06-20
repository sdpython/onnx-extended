#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

struct CustomGemmKernel {
  CustomGemmKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  void set(const std::vector<int64_t> &shape_a,
           const std::vector<int64_t> &shape_b, int &M, int &N, int &K,
           int &lda, int &ldb, int &ldd) const;

  float alpha_;
  // float beta_;
  bool transA_;
  bool transB_;
  bool fastAccumulationMode_;
  bool row_major_;
  int64_t smCount_;
  cublasComputeType_t computeType_;
};

struct CustomGemmOpFloat
    : Ort::CustomOpBase<CustomGemmOpFloat, CustomGemmKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080

struct CustomGemmOpFloat8E4M3FN
    : Ort::CustomOpBase<CustomGemmOpFloat8E4M3FN, CustomGemmKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

#endif

} // namespace ortops
