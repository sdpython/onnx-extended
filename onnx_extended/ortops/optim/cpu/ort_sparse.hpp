#pragma once

#include "ort_sparse.h"
#include "common/sparse_tensor.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

// Regressor

template <typename T>
void *DenseToSparse<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<DenseToSparseKernel<T>>(api, info).release();
};

template <> const char *DenseToSparse<float>::GetName() const { return "DenseToSparse"; };

template <typename T> const char *DenseToSparse<T>::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

template <typename T> size_t DenseToSparse<T>::GetInputTypeCount() const { return 1; };

template <>
ONNXTensorElementDataType DenseToSparse<float>::GetInputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

template <typename T> size_t DenseToSparse<T>::GetOutputTypeCount() const { return 1; };

template <>
ONNXTensorElementDataType DenseToSparse<float>::GetOutputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

/////////
// Kernel
/////////

template <typename T>
DenseToSparseKernel<T>::DenseToSparseKernel(const OrtApi & /* api */, const OrtKernelInfo * /* info */) {}

template <typename T> void DenseToSparseKernel<T>::Compute(OrtKernelContext *context) {

  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  const T *X = input_X.GetTensorData<T>();
  std::vector<int64_t> dimensions_in = input_X.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions_in.size() == 2, "DenseToSparse only allows 2D inputs.");
  int64_t n_rows = dimensions_in[0];
  int64_t n_cols = dimensions_in[1];

  int64_t n_elements = n_rows * n_cols;
  uint32_t n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (X[i] != 0)
      ++n_els;
  }
  std::size_t size_float = onnx_sparse::sparse_struct::size_float(n_els, 1);

  std::vector<int64_t> dimensions_out{static_cast<int64_t>(size_float)};
  Ort::UnownedValue output = ctx.GetOutput(0, dimensions_out);
  T *out = output.GetTensorMutableData<T>();

  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)out;
  sp->set(dimensions_in, n_els, 1);
  uint32_t *indices = sp->indices();
  float *values = sp->values();

  // The implementation could be parallelized.
  n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (X[i] != 0) {
      indices[n_els] = i;
      values[n_els] = X[i];
      ++n_els;
    }
  }
}

} // namespace ortops
