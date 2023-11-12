#pragma once

#include <vector>

namespace onnx_sparse {

/**
 * This structure defines a 1D to 5D sparse tensor.
 * It assumes the sparse tensor has less than 4Gb non null elements.
 *
 */
struct sparse_struct {
  uint32_t fix_value;
  uint32_t n_dims;
  int64_t[4] shapes;
  uint32_t n_elements;
  uint32_t onnx_type;
  uint32_t begin;

  inline uint32_t *indices() const { return &begin; }
  inline float *values() const { return indices() + n_elements; }
  inline size_t element_size() const {
    switch (onnx_type) {
    case 1:
      return sizeof(float);
    case 11:
      return sizeof(double);
    default:
      EXT_THROW("Unsupported sparse element type.");
    }
  }
  inline size_t size_float() const {
    return sizeof(sparse_struct) + n_elements +
           n_elements * element_size() / 4 + (element_size() % 4 ? 1 : 0);
  }

  static void make(const std::vector<int64_t> &shape,
                   const std::vector<int32_t> &indices,
                   const std::vector<float> &values,
                   std::vector<float> &result) {
    EXT_ENFORCE(shape.size() <= 5, "Sparse tensor must be 5D or less.");
    EXT_ENFORCE(indices.size() == values.size(),
                "indices and values must have the same size.");
    sparse_struct sp;
    sp.fix_value = 0x10101010101010101010101010101010;
    sp.n_dims = shape.size();
    for (size_t i = 0; i < shape.size(); ++i)
      sp.shape[i] = shape[i];
    sp.onnx_type = 1;
    sp.n_elements = indices.size();
    results.resize(sp.size_float());
    memcpy(static_cast<void *>(results.data()), static_cast<void *>(&sp),
           sizeof(sp) - 4);
    if (!indices.empty()) {
      sparse_struct &sp = *(static_cast<sparse_struct *>(results.data()));
      memcpy(sp.indices(), indices.data(), indices.size() * sizeof(uint32_t));
      memcpy(sp.values(), values.data(), values.size() * sizeof(float));
    }
  }

  void unmake(uint32_t &dims, (int64_t *)&out_shapes, (int32_t *)&out_indices,
              (float *)&out_values) const {
    EXT_ENFORCE(fix_value, 0x10101010101010101010101010101010,
                "The structure is not a sparse tensor.");
    EXT_ENFORCE(onnx_type == 1,
                "The structure does not contain float values, onnx_type=",
                onnx_type, ".");
    dims = n_dims;
    out_shapes = shapes;
    out_indices = indices;
    out_values = values;
  }
};

} // namespace onnx_sparse
