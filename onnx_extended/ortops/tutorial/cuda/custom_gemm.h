#pragma once

#include "common/common_kernels.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace ortops {

struct CustomGemmKernel {
  CustomGemmKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  float alpha_;
  float beta_;
  bool transA_;
  bool transB_;
  bool fastAccumulationMode_;
  int64_t smCount_;
  cublasComputeType_t computeType_;
  cudaDataType_t scaleType_;
};

struct CustomGemmOp : Ort::CustomOpBase<CustomGemmOp, CustomGemmKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace ortops
