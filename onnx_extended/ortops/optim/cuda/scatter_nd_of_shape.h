#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include "scatter_nd_of_shape_common.h"
#include <cuda_runtime.h>

namespace ortops {

template <typename T> struct ScatterNDOfShapeKernel {
  ScatterNDOfShapeKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  void ComputeNone(cudaStream_t &stream, const std::vector<int64_t> &input_shape,
                   const std::vector<int64_t> &indices_shape, T *output_data,
                   const int64_t *indices_data, const T *updates_data) const;
  void ComputeOptimize(cudaStream_t &stream, const std::vector<int64_t> &input_shape,
                       const std::vector<int64_t> &indices_shape, T *output_data,
                       const int64_t *indices_data, const T *updates_data) const;

  Reduction reduction_;
  Strategy strategy_;
  int maxThreadPerBlock_;
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
  OrtMemType GetInputMemoryType(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

} // namespace ortops
