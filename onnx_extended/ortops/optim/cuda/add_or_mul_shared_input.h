#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

template <typename T, bool addition> struct AddOrMulSharedInputKernel {
  AddOrMulSharedInputKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

template <typename T, bool addition>
struct AddOrMulSharedInputOp
    : Ort::CustomOpBase<AddOrMulSharedInputOp<T, addition>, AddOrMulSharedInputKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddOrMulSharedInputOp<T, addition>, AddOrMulSharedInputKernel<T, addition>>
      parent_type;
  AddOrMulSharedInputOp() : parent_type() {}
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
