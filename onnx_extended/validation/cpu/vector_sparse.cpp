#include "vector_sparse.h"
#include "common/c_op_helpers.h"

#include <omp.h>
#include <chrono>
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

//////////////////////////////////////////////////
// Function to evaluate access to sparse structure
//////////////////////////////////////////////////

static void fill_random_indices(int n_row, int n_col, int nr, int ntimes,
                                std::vector<int64_t> &res) {
  res.resize(n_row * nr * ntimes);
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = std::rand() % n_col;
  }
}

static void dense_to_sparse(const float *v, int64_t n_rows, int64_t n_cols,
                            std::vector<float> &sparse_tensor) {
  int64_t n_elements = n_rows * n_cols;
  // input was given as dense.
  uint32_t n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (v[i] != 0)
      ++n_els;
  }
  std::size_t size_float = onnx_sparse::sparse_struct::size_float(n_els, 1);

  sparse_tensor.resize(size_float);
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)sparse_tensor.data();
  std::vector<int64_t> dims{n_rows, n_cols};
  sp->set(dims, n_els, 1);
  uint32_t *indices = sp->indices();
  float *values = sp->values();

  n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (v[i] != 0) {
      indices[n_els] = i;
      values[n_els] = v[i];
      ++n_els;
    }
  }
}

static void sparse_to_dense(const std::vector<float> &sparse_tensor, std::vector<float> &res) {
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)sparse_tensor.data();
  EXT_ENFORCE(sp->n_dims == 2, "Expected a 2D tensor.");
  res.resize(sp->shape[0] * sp->shape[1]);
  std::fill(res.begin(), res.end(), (float)0);
  const uint32_t *indices = sp->indices();
  const float *values = sp->values();
  for (std::size_t i = 0; i < sp->n_elements; ++i) {
    res[indices[i]] = values[i];
  }
}

static std::tuple<double, double, double>
_test_sparse_dense(std::vector<int64_t> &pos, const std::vector<float> &sparse_tensor,
                   int random, int ntimes, int repeat) {

  double performance = 0;
  double init = 0;

  // initialization
  onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)sparse_tensor.data();
  std::vector<float> dense;
  int64_t n_rows = static_cast<int64_t>(sp->shape[0]);
  int64_t n_cols = static_cast<int64_t>(sp->shape[1]);

  std::vector<float> res(n_rows);
  float sum = 0;
  for (int r = 0; r < repeat; ++r) {
    auto time0 = std::chrono::high_resolution_clock::now();
    sparse_to_dense(sparse_tensor, dense);
    const float *v = dense.data();

    init += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
                .count();

    time0 = std::chrono::high_resolution_clock::now();

    // loop to test
    sum = 0;
    for (int64_t row = 0; row < n_rows; ++row) {
      for (int ti = 0; ti < ntimes; ++ti) {
        const float *pf = v + row * n_cols;
        const int64_t *pi = pos.data() + row * random * ntimes + ti * random;
        for (int c = 0; c < random; ++c) {
          sum += pf[pi[c]];
        }
        res[row] = sum;
      } // end of the loop to test
    }

    performance +=
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
            .count();
  }
  return std::tuple<double, double, double>(init / repeat, performance / repeat,
                                            static_cast<double>(sum));
}

static std::tuple<double, double, double>
_test_sparse_sparse(std::vector<int64_t> &pos, int64_t n_elements, const float *v,
                    int64_t n_rows, int64_t n_cols, int random, int ntimes, int repeat) {
  const onnx_sparse::sparse_struct *sp = (onnx_sparse::sparse_struct *)v;
  EXT_ENFORCE(sp->n_dims == 2, "Number of dimensions should be two, not ", sp->n_dims, ".");
  EXT_ENFORCE(sp->shape[0] == n_rows, "Mismatch dimension 0, ", sp->shape[0], "!=", n_rows,
              ".");
  EXT_ENFORCE(sp->shape[1] == n_cols, "Mismatch dimension 1, ", sp->shape[1], "!=", n_cols,
              ".");
  EXT_ENFORCE(n_elements == static_cast<int64_t>(sp->size_float()),
              "Unexpected number of elements, ", n_elements,
              "!=", static_cast<int64_t>(sp->size_float()), ".");

  double performance = 0;
  double init = 0;
  std::vector<float> res(n_rows);
  std::vector<uint32_t> row_indices;
  std::vector<uint32_t> element_indices;
  float sum = 0;

  // computation
  for (int r = 0; r < repeat; ++r) {
    // initialisation
    auto time0 = std::chrono::high_resolution_clock::now();
    sp->csr(row_indices, element_indices);
    init += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
                .count();

    time0 = std::chrono::high_resolution_clock::now();

    // loop to test
    sum = 0;
    for (int64_t row = 0; row < n_rows; ++row) {

      float *values = sp->values();
      uint32_t *root = &(element_indices[0]);
      auto begin = element_indices.data() + row_indices[row];
      auto end = element_indices.data() + row_indices[row + 1];
      for (int ti = 0; ti < ntimes; ++ti) {
        const int64_t *pi = pos.data() + row * random * ntimes + ti * random;
        for (int c = 0; c < random; ++c) {
          auto it = std::lower_bound(begin, end, static_cast<uint32_t>(pi[c]));
          sum += (it != end && pi[c] == *it) ? values[it - root] : static_cast<float>(0);
        }
      }
      res[row] = sum;
    } // end of the loop to test
    performance +=
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
            .count();
  }
  return std::tuple<double, double, double>(init / repeat, performance / repeat,
                                            static_cast<double>(sum));
}

std::vector<std::tuple<double, double, double>> evaluate_sparse(const float *v, int64_t n_rows,
                                                                int64_t n_cols, int random,
                                                                int ntimes, int repeat,
                                                                int test) {
  std::vector<std::tuple<double, double, double>> res;
  std::vector<int64_t> pos;
  fill_random_indices(n_rows, n_cols, random, ntimes, pos);
  std::vector<float> sparse_tensor;
  dense_to_sparse(v, n_rows, n_cols, sparse_tensor);
  if (test & 1) {
    auto r = _test_sparse_dense(pos, sparse_tensor, random, ntimes, repeat);
    res.push_back(r);
  }
  if (test & 2) {
    auto r = _test_sparse_sparse(pos, sparse_tensor.size(), sparse_tensor.data(), n_rows,
                                 n_cols, random, ntimes, repeat);
    res.push_back(r);
  }
  EXT_ENFORCE(res.size() > 0, "No test was run, test=", test, ".");
  return res;
}

} // namespace validation
