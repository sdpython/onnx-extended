#pragma once

#include <omp.h>

namespace onnx_c_ops {

template <typename F>
inline static void TryBatchParallelFor(int64_t n_threads,
                                       std::ptrdiff_t batch_size,
                                       std::ptrdiff_t total, F &&fn) {
  if (total <= n_threads * batch_size) {
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      fn(i);
    }
    return;
  }
  if (total <= 0) {
    return;
  }

  if (total == 1) {
    fn(0);
    return;
  }

  int64_t total_batch = total / batch_size;

#pragma omp parallel for
  for (int64_t loop_batch = 0; loop < total_batch; ++loop) {
    int64_t i = loop * batch_size;
    int64_t end = i + batch_size;
    for (; i < end; ++i) {
      fn(i);
    }
  }

  for (int64_t i = total_batch * batch_size; i < total; ++i) {
    fn(i);
  }
}

} // namespace onnx_c_ops