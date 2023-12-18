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

static std::tuple<double, double> _test_sparse_dense(int64_t n_elements, const float *v,
                                                     int64_t n_rows, int64_t n_cols, int random,
                                                     int number, int repeat) {

  EXT_ENFORCE(n_elements == n_rows * n_cols, "Dimension mismatch, n_elements=", n_elements,
              " n_rows * n_cols=", n_rows * n_cols, ".");
  double performance = 0;
  std::vector<int64_t> pos;
  std::vector<float> res(n_rows);
  fill_random_indices(n_rows, n_cols, random, pos);
  for (int r = 0; r < repeat; ++r) {
    auto time0 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < number; ++n) {
      // loop to test
      for (int64_t row = 0; row < n_rows; ++row) {
        float sum = 0;
        const float *pf = v + row * n_cols;
        const int64_t *pi = pos.data() + row * random;
        for (int c = 0; c < random; ++c) {
          sum += pf[pi[c]];
        }
        res[row] = sum;
      }
    }

    performance +=
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
            .count();
  }
  return std::tuple<double, double>(0, performance / repeat);
}

static std::tuple<double, double> _test_sparse_sparse(int64_t n_elements, const float *v,
                                                      int64_t n_rows, int64_t n_cols,
                                                      int random, int number, int repeat) {
  EXT_ENFORCE(n_elements > 0);
  EXT_ENFORCE(v != nullptr);
  EXT_ENFORCE(n_rows > 0);
  EXT_ENFORCE(n_cols > 0);
  EXT_ENFORCE(random > 0);
  for (int r = 0; r < repeat; ++r) {
    for (int n = 0; n < number; ++n) {
    }
  }
  return std::tuple<double, double>(0, 0);
}

std::tuple<double, double> evaluate_sparse(int64_t n_elements, const float *v, int64_t n_rows,
                                           int64_t n_cols, int random, int number, int repeat,
                                           bool dense) {
  return dense ? _test_sparse_dense(n_elements, v, n_rows, n_cols, random, number, repeat)
               : _test_sparse_sparse(n_elements, v, n_rows, n_cols, random, number, repeat);
}

} // namespace validation
