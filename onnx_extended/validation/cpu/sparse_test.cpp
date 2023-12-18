#pragma once

#include "sparse_test.h"
#include "common/sparse_struct.h"
#include <chrono>

namespace validation {

static void random(int n_row, int n_col, int nr, std::vector<int64_t> &res) {
  res.resize(n_row, nr);
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = std::rand() % n_col;
  }
}

static std::tuple<double, double> _test_sparse_dense(int64_t, const float *v, int64_t n_row,
                                                     int64_t n_col, int n_random, int n_number,
                                                     int n_repeat) {

  double performance = 0;
  std::vector<int64_t> pos;
  std::vector<float> res(n_rows);
  random(n_row, n_col, nr, pos);
  for (int r = 0; r < n_repeat; ++r) {
    auto time0 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < n_number; ++n) {
      // loop to test
      for (int64_t row = 0; row < n_row; ++row) {
        float sum = 0;
        const float *pf = v + row * n_col;
        const int64_t *pi = pos.data() + row * nr;
        for (int c = 0; c < n_random; ++c) {
          sum += pf[pi[c]];
        }
        n_rows[row] = sum;
      }
    }

    performance +=
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
            .count();
  }
  return std::tuple<double, double>(0, performance / n_repeat);
}

static std::tuple<double, double> _test_sparse_sparse(int64_t n, const float *v, int64_t n_row,
                                                      int64_t n_col, int n_random, int n_number,
                                                      int n_repeat) {
  for (int r = 0; r < n_repeat; ++r) {
    for (int n = 0; n < n_number; ++n) {
    }
  }
  return std::tuple<double, double>(0, 0);
}

std::tuple<double, double> evaluate_sparse(int64_t n, const float *v, int64_t n_row,
                                           int64_t n_col, int n_random, int n_number,
                                           int n_repeat, bool dense) {
  return dense ? _test_sparse_dense(n, col, v, n_random, n_number, n_repeat)
               : _test_sparse_sparse(n, col, v, n_random, n_number, n_repeat);
}

} // namespace validation
