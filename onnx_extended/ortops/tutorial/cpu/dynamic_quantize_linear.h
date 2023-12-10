#pragma once

#include "common/common_kernels.h"

namespace ortops {

struct DynamicQuantizeLinearKernel {
  DynamicQuantizeLinearKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  template <typename T>
  void ComputeInternal(int64_t n_elements, const T *input, uint8_t *output, float &scale,
                       uint8_t &zero_point);

  int64_t to_;
};

struct DynamicQuantizeLinearOp
    : Ort::CustomOpBase<DynamicQuantizeLinearOp, DynamicQuantizeLinearKernel> {
  typedef Ort::CustomOpBase<DynamicQuantizeLinearOp, DynamicQuantizeLinearKernel> parent_type;
  DynamicQuantizeLinearOp(ONNXTensorElementDataType input_type,
                          ONNXTensorElementDataType quant_type)
      : parent_type(), input_type_(input_type), quant_type_(quant_type) {}

  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const noexcept;
  const char *GetName() const noexcept;
  const char *GetExecutionProviderType() const noexcept;

  std::size_t GetInputTypeCount() const noexcept;
  ONNXTensorElementDataType GetInputType(std::size_t index) const noexcept;

  std::size_t GetOutputTypeCount() const noexcept;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;

private:
  ONNXTensorElementDataType input_type_;
  ONNXTensorElementDataType quant_type_;
};

} // namespace ortops
