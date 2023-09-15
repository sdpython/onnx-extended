#pragma once

#include <string>
#include <unordered_map>

namespace cuda_example {

struct BenchmarkGemm {
  int64_t N;
  double workspace_new;
  double workspace_free;
  double stream_create;
  double stream_destroy;
  double setup;
  double clean;
  double gemm;
  double gemm_in;
  double gemm_sync;
  double total;
  BenchmarkGemm();
  void zero();
  void to_map(std::unordered_map<std::string, double> &bench);
};

std::unordered_map<std::string, double>
gemm_benchmark_test(int test, int N, int m, int n, int k, int lda, int ldb, int ldd);

} // namespace cuda_example
