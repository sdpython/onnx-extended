#pragma once

#include "common/common_kernels.h"

namespace ortops {

struct CustomTreeAssemblyKernel {
  CustomTreeAssemblyKernel(const OrtApi &api, const OrtKernelInfo *info, bool classifier);
  void Compute(OrtKernelContext *context);
  ~CustomTreeAssemblyKernel();

  bool classifier_;
  std::string assembly_name_;
  /* TreebeardSORunner */ void *assembly_runner_;
};

struct CustomTreeAssemblyOp
    : Ort::CustomOpBase<CustomTreeAssemblyOp, CustomTreeAssemblyKernel> {
  typedef Ort::CustomOpBase<CustomTreeAssemblyOp, CustomTreeAssemblyKernel> parent_type;
  CustomTreeAssemblyOp(bool classifier) : parent_type(), classifier_(classifier) {}
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;

private:
  bool classifier_;
};

} // namespace ortops
