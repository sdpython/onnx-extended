#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

template <typename T> struct MulMulSigmoidKernel {
  MulMulSigmoidKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

template <typename T>
struct MulMulSigmoidOp : Ort::CustomOpBase<MulMulSigmoidOp<T>, MulMulSigmoidKernel<T>> {
  typedef Ort::CustomOpBase<MulMulSigmoidOp<T>, MulMulSigmoidKernel<T>> parent_type;
  MulMulSigmoidOp() : parent_type() {}
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

} // namespace ortops
