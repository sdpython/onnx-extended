#include "vector_sum.h"

#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark

namespace validation {

float vector_sum(int nc, const std::vector<float> &values, bool by_rows) {
  float total = 0;
  if (by_rows) {
    /*
    for(size_t i=0; i < values.size(); ++i) {
        total += values[i];
    }
    */
    int nl = values.size() / nc;
    for (size_t i = 0; i < nl; ++i) {
      for (size_t j = 0; j < nc; ++j) {
        total += values[i * nc + j];
      }
    }
  } else {
    int nl = values.size() / nc;
    for (size_t j = 0; j < nc; ++j) {
      for (size_t i = 0; i < nl; ++i) {
        total += values[i * nc + j];
      }
    }
  }
  return total;
}

float vector_sum(int nl, int nc, const float* values, bool by_rows) {
  float total = 0;
  if (by_rows) {
    for (size_t i = 0; i < nl; ++i) {
      for (size_t j = 0; j < nc; ++j) {
        total += values[i * nc + j];
      }
    }
  } else {
    for (size_t j = 0; j < nc; ++j) {
      for (size_t i = 0; i < nl; ++i) {
        total += values[i * nc + j];
      }
    }
  }
  return total;
}

float vector_sum_array(int nc, const py_array_float &values_array,
                       bool by_rows) {
  const float *values = values_array.data(0);

  float total = 0;
  if (by_rows) {
    /*
    for(size_t i=0; i < values.size(); ++i) {
        total += values[i];
    }
    */
    int nl = values_array.size() / nc;
    for (size_t i = 0; i < nl; ++i) {
      for (size_t j = 0; j < nc; ++j) {
        total += values[i * nc + j];
      }
    }
  } else {
    int nl = values_array.size() / nc;
    for (size_t j = 0; j < nc; ++j) {
      for (size_t i = 0; i < nl; ++i) {
        total += values[i * nc + j];
      }
    }
  }
  return total;
}

float vector_sum_array_parallel(int nc, const py_array_float &values_array,
                                bool by_rows) {
  int n_threads = omp_get_max_threads();
  const float *values = values_array.data(0);
  std::vector<float> totals(n_threads, 0);

  if (by_rows) {
    /*
    for(size_t i=0; i < values.size(); ++i) {
        total += values[i];
    }
    */
    int nl = values_array.size() / nc;
#pragma omp parallel for
    for (int i = 0; i < nl; ++i) {
      auto th = omp_get_thread_num();
      for (size_t j = 0; j < nc; ++j) {
        totals[th] += values[i * nc + j];
      }
    }
  } else {
    int nl = values_array.size() / nc;
#pragma omp parallel for
    for (int j = 0; j < nc; ++j) {
      auto th = omp_get_thread_num();
      for (size_t i = 0; i < nl; ++i) {
        totals[th] += values[i * nc + j];
      }
    }
  }

  for (size_t i = 1; i < totals.size(); ++i) {
    totals[0] += totals[i];
  }
  return totals[0];
}

float vector_sum_array_avx(int nc, const py_array_float &values_array) {
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
  const float *values = values_array.data(0);
  float total = 0;
  float buffer[8];

  int nl = values_array.size() / nc;
  for (int i = 0; i < nl; ++i) {
    __m256 t = _mm256_set1_ps(0);
    size_t j = 0;
    for (; j < nc; j += 8) {
      __m256 vec = _mm256_loadu_ps(&(values[i * nc + j]));
      t = _mm256_add_ps(vec, t);
    }
    _mm256_storeu_ps(buffer, t);
#pragma unroll
    for (size_t k = 0; k < 8; ++k) {
      total += buffer[k];
    }
    for (; j < nc; ++j) {
      total += values[i * nc + j];
    }
  }
  return total;
}

float vector_sum_array_avx_parallel(int nc,
                                    const py_array_float &values_array) {
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
  int n_threads = omp_get_max_threads();
  const float *values = values_array.data(0);
  std::vector<float> totals(n_threads, 0);

  int nl = values_array.size() / nc;
#pragma omp parallel for
  for (int i = 0; i < nl; ++i) {
    float buffer[8];
    __m256 t = _mm256_set1_ps(0);
    size_t j = 0;
    for (; j < nc; j += 8) {
      __m256 vec = _mm256_loadu_ps(&(values[i * nc + j]));
      t = _mm256_add_ps(vec, t);
    }
    _mm256_storeu_ps(buffer, t);
    auto th = omp_get_thread_num();
#pragma unroll
    for (size_t k = 0; k < 8; ++k) {
      totals[th] += buffer[k];
    }
    for (; j < nc; ++j) {
      totals[th] += values[i * nc + j];
    }
  }
  for (size_t i = 1; i < totals.size(); ++i) {
    totals[0] += totals[i];
  }
  return totals[0];
}

} // namespace validation
