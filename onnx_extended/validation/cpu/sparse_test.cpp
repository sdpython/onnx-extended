#include "sparse_test.h"
#include "common/sparse_tensor.h"
#include <chrono>

namespace validation {

static void fill_random_indices(int n_row, int n_col, int nr, std::vector<int64_t> &res) {
  res.resize(n_row * nr);
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
  for (std::size_t i = 0; i < sp->n_elements; ++i) {
    res[sp->indices()[i]] = sp->values()[i];
  }
}

static std::tuple<double, double, double>
_test_sparse_dense(std::vector<int64_t> &pos, const std::vector<float> &sparse_tensor,
                   int random, int repeat) {

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
      const float *pf = v + row * n_cols;
      const int64_t *pi = pos.data() + row * random;
      for (int c = 0; c < random; ++c) {
        sum += pf[pi[c]];
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

static std::tuple<double, double, double>
_test_sparse_sparse(std::vector<int64_t> &pos, int64_t n_elements, const float *v,
                    int64_t n_rows, int64_t n_cols, int random, int repeat) {
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
      const int64_t *pi = pos.data() + row * random;
      for (int c = 0; c < random; ++c) {
        auto it = std::lower_bound(begin, end, static_cast<uint32_t>(pi[c]));
        sum += (it != end && pi[c] == *it) ? values[it - root] : static_cast<float>(0);
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
                                                                int repeat, int test) {
  std::vector<std::tuple<double, double, double>> res;
  std::vector<int64_t> pos;
  fill_random_indices(n_rows, n_cols, random, pos);
  std::vector<float> sparse_tensor;
  dense_to_sparse(v, n_rows, n_cols, sparse_tensor);
  if (test & 1) {
    auto r = _test_sparse_dense(pos, sparse_tensor, random, repeat);
    res.push_back(r);
  }
  if (test & 2) {
    auto r = _test_sparse_sparse(pos, sparse_tensor.size(), sparse_tensor.data(), n_rows,
                                 n_cols, random, repeat);
    res.push_back(r);
  }
  EXT_ENFORCE(res.size() > 0, "No test was run, test=", test, ".");
  return res;
}

} // namespace validation
