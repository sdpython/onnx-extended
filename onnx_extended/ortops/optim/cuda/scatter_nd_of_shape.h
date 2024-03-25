#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

enum class Reduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

template <typename T> struct ScatterNDOfShapeKernel {
  ScatterNDOfShapeKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  Reduction reduction_;
};

template <typename T>
struct ScatterNDOfShapeOp
    : Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> {
  typedef Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> parent_type;
  ScatterNDOfShapeOp() : parent_type() {}
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
