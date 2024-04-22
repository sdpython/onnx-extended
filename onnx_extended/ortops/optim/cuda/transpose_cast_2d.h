#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

struct Transpose2DCastKernel {
  Transpose2DCastKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

struct Transpose2DCastOp : Ort::CustomOpBase<Transpose2DCastOp, Transpose2DCastKernel> {
  typedef Ort::CustomOpBase<Transpose2DCastOp, Transpose2DCastKernel> parent_type;
  Transpose2DCastOp(ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type)
      : parent_type() {
    input_type_ = input_type;
    output_type_ = output_type;
  }
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
  ONNXTensorElementDataType input_type_;
  ONNXTensorElementDataType output_type_;
};

} // namespace ortops
