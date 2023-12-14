#include "vector_sparse.h"
#include "common/c_op_helpers.h"

#include <omp.h>
#if __cpluscplus >= 202002L
#include <span>
#else
#include "common/simple_span.h"
#endif

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark

namespace validation {

py_array_float dense_to_sparse_struct(const py_array_float &v) {
  std::vector<int64_t> dims;
  arrayshape2vector(dims, v);
  py::buffer_info brv = v.request();
  float *pv = static_cast<float *>(brv.ptr);

  uint32_t n_elements = 0;
  std::size_t size_v = static_cast<std::size_t>(v.size());
  for (std::size_t i = 0; i < size_v; ++i) {
    if (pv[i] != 0)
      ++n_elements;
  }
  std::size_t size_float = onnx_sparse::sparse_struct::size_float(n_elements, 1);

  std::vector<int64_t> out_dims{static_cast<int64_t>(size_float)};
  py_array_float result = py::array_t<float>(out_dims);
  py::buffer_info br = result.request();
  float *pr = static_cast<float *>(br.ptr);

  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)pr;
  sp->set(dims, n_elements, 1);
  uint32_t *indices = sp->indices();
  float *values = sp->values();

  n_elements = 0;
  for (std::size_t i = 0; i < size_v; ++i) {
    if (pv[i] != 0) {
      indices[n_elements] = i;
      values[n_elements] = pv[i];
      ++n_elements;
    }
  }
  return result;
}

py_array_float sparse_struct_to_dense(const py_array_float &v) {
  py::buffer_info br = v.request();
  float *pr = static_cast<float *>(br.ptr);
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)pr;
  uint32_t *indices;
  float *values;
  uint32_t n_dims;
  const int64_t *shape;
  uint32_t n_elements;
  sp->unmake(n_dims, n_elements, shape, indices, values);
#if __cpluscplus >= 202002L
  std::span<int64_t> out_shape((int64_t *)shape, n_dims);
#else
  std_::span<int64_t> out_shape((int64_t *)shape, n_dims);
#endif

  py_array_float result = py::array_t<float>(out_shape);
  py::buffer_info brout = result.request();
  pr = static_cast<float *>(brout.ptr);
  int64_t size = onnx_c_ops::flattened_dimension(out_shape);
  std::memset(pr, 0, size * sizeof(float));
  for (uint32_t pos = 0; pos < n_elements; ++pos) {
    pr[indices[pos]] = values[pos];
  }
  return result;
}

py::tuple sparse_struct_indices_values(const py_array_float &v) {
  py::buffer_info br = v.request();
  float *pr = static_cast<float *>(br.ptr);
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)pr;

  py_array_uint32 rind =
      py::array_t<uint32_t>(std::vector<int64_t>{static_cast<int64_t>(sp->n_elements)});
  py::buffer_info brout = rind.request();
  std::memcpy(brout.ptr, sp->indices(), sp->n_elements * sizeof(uint32_t));

  py_array_float rval =
      py::array_t<float>(std::vector<int64_t>{static_cast<int64_t>(sp->n_elements)});
  py::buffer_info broutv = rval.request();
  std::memcpy(broutv.ptr, sp->values(), sp->n_elements * sizeof(float));

  return py::make_tuple(rind, rval);
}

py::list sparse_struct_to_maps(const py_array_float &v) {
  py::buffer_info br = v.request();
  float *pr = static_cast<float *>(br.ptr);
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)pr;
  std::vector<std::map<uint32_t, float>> maps;
  sp->to_maps(maps);
  py::list res;
  for (auto it : maps) {
    py::dict d;
    for (auto pair : it) {
      d[py::int_(pair.first)] = pair.second;
    }
    res.append(d);
  }
  return res;
}

py::tuple sparse_struct_to_csr(const py_array_float &v) {
  py::buffer_info br = v.request();
  float *pr = static_cast<float *>(br.ptr);
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)pr;
  std::vector<uint32_t> row_indices;
  std::vector<uint32_t> element_indices;
  sp->csr(row_indices, element_indices);

  // The copy should be avoided but this function is more useful in C.
  py_array_uint32 arow =
      py::array_t<uint32_t>(std::vector<int64_t>{static_cast<int64_t>(row_indices.size())});
  py::buffer_info brout = arow.request();
  std::memcpy(brout.ptr, row_indices.data(), row_indices.size() * sizeof(uint32_t));

  py_array_uint32 aels =
      py::array_t<uint32_t>(std::vector<int64_t>{static_cast<int64_t>(element_indices.size())});
  py::buffer_info broute = aels.request();
  std::memcpy(broute.ptr, element_indices.data(), element_indices.size() * sizeof(uint32_t));
  return py::make_tuple(arow, aels);
}

} // namespace validation
