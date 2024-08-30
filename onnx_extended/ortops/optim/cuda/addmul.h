#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

template <typename T, bool addition> struct AddMulKernel {
  AddMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
  private:
    // If true, the operator assumes there are 4 dimensions and the two middle ones are switched.
    bool switch_middle_axis_;
};

template <typename T, bool addition>
struct AddMulOp : Ort::CustomOpBase<AddMulOp<T, addition>, AddMulKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddMulOp<T, addition>, AddMulKernel<T, addition>> parent_type;
  AddMulOp() : parent_type() {}
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
