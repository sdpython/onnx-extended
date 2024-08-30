#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

struct MatxMatMulKernel {
  MatxMatMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

struct MatxMatMulOp : Ort::CustomOpBase<MatxMatMulOp, MatxMatMulKernel> {
  typedef Ort::CustomOpBase<MatxMatMulOp, MatxMatMulKernel> parent_type;
  MatxMatMulOp(const char *op_name, ONNXTensorElementDataType dtype) : parent_type() {
    op_name_ = op_name;
    dtype_ = dtype;
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
  const char *op_name_;
  ONNXTensorElementDataType dtype_;
};

} // namespace ortops
