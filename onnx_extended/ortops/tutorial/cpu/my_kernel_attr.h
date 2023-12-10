#pragma once

#include "common/common_kernels.h"

namespace ortops {

struct MyCustomKernelWithAttributes {
  MyCustomKernelWithAttributes(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  std::string att_string;
  float att_float;
  int64_t att_int64;
  std::vector<double> att_tensor_double;
};

struct MyCustomOpWithAttributes
    : Ort::CustomOpBase<MyCustomOpWithAttributes, MyCustomKernelWithAttributes> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

} // namespace ortops
