#pragma once

#include <string>
#include <unordered_map>
#include "cuda_tensor.cuh"

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

std::unordered_map<std::string, double> gemm_benchmark_test(int test, int N, int m, int n,
                                                            int k, int lda, int ldb, int ldd);

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_type;

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n, int k, int lda,
                 int ldb, int ldd, cublasOperation_t transa, cublasOperation_t transb,
                 bool grad, void *workspace, std::size_t workspaceSize, bool accumulate,
                 bool use_split_accumulator, int math_sm_count,
                 cublasComputeType_t gemm_compute_type, cudaStream_t stream, time_type &begin,
                 time_type &heuristic, time_type &end, time_type &end2, int &i_epilogue,
                 int &i_compute_type, int &i_algo);                                                            

} // namespace cuda_example
