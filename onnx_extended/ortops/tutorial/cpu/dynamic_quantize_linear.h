#pragma once

#include "common/common_kernels.h"

namespace ortops {

struct DynamicQuantizeLinearKernel {
  DynamicQuantizeLinearKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  template <typename T>
  void ComputeInternal(int64_t n_elements, const T *input, uint8_t *output,
                       float &scale, uint8_t &zero_point);

  int64_t to_;
};

struct DynamicQuantizeLinearOp
    : Ort::CustomOpBase<DynamicQuantizeLinearOp, DynamicQuantizeLinearKernel> {
  void *CreateKernel(const OrtApi &api,
                     const OrtKernelInfo *info) const noexcept;
  const char *GetName() const noexcept;
  const char *GetExecutionProviderType() const noexcept;

  size_t GetInputTypeCount() const noexcept;
  ONNXTensorElementDataType GetInputType(size_t index) const noexcept;
  constexpr OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t index) const noexcept;

  size_t GetOutputTypeCount() const noexcept;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  constexpr OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t index) const;
};

} // namespace ortops
