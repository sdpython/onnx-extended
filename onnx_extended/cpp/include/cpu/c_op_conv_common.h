#pragma once

#include <Eigen/Dense>

using namespace Eigen;

namespace onnx_c_ops {

// The function adds value to C, assuming this array
// was initialized.
template <typename NTYPE>
void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, NTYPE alpha,
          const NTYPE *A, const NTYPE *B, NTYPE beta, NTYPE *C) {
#if defined(__APPLE__)
  // link issues on apple, "___kmpc_fork_call", referenced from:
  if (transA) {
    if (transB) {
    } else {
      // a A B + b C, dimension = M * N
      NTYPE *begin;
      NTYPE val;
      NTYPE val0;
      std::size_t i, j, k, maxc = 0;
      const NTYPE *pA, *pB;
      for (i = 0, begin = C; i < M; ++i) {
        for (j = 0; j < N; ++j, ++begin) {
          val0 = *begin * beta;
          val = 0;
          pA = A + i;
          pB = B + j;
          for (k = K; k > 0; --k, pA += K, pB += N)
            val += *pA * *pB;
          *begin = val0 + val * alpha;
          maxc = maxc > (std::size_t)(begin - C) ? maxc : (std::size_t)(begin - C);
          if (maxc > M * N)
            throw std::invalid_argument("gemm10: maxc > M * N");
        }
      }
      return;
    }
  } else {
    if (transB) {
    } else {
      // a A B + b C, dimension = M * N
      NTYPE *begin;
      NTYPE val;
      NTYPE val0;
      std::size_t i, j, k, maxc = 0;
      const NTYPE *pA, *pB;
      for (i = 0, begin = C; i < M; ++i) {
        for (j = 0; j < N; ++j, ++begin) {
          val0 = *begin * beta;
          val = 0;
          pA = A + i * K;
          pB = B + j;
          for (k = K; k > 0; --k, ++pA, pB += N)
            val += *pA * *pB;
          *begin = val0 + val * alpha;
          maxc = maxc > (std::size_t)(begin - C) ? maxc : (std::size_t)(begin - C);
          if (maxc > M * N)
            throw std::invalid_argument("gemm00: maxc > M * N");
        }
      }
      return;
    }
  }
#else
  typedef Map<Matrix<NTYPE, Dynamic, Dynamic, RowMajor>> matrixdd_row;
  typedef Map<Matrix<NTYPE, Dynamic, Dynamic, ColMajor>> matrixdd_col;
  matrixdd_row mc(C, M, N);
  if (beta != 1)
    mc *= beta;
  if (transA) {
    matrixdd_col ma((NTYPE *)A, M, K);
    if (transB) {
      matrixdd_col mb((NTYPE *)B, K, N);
      if (alpha != 1)
        mc.noalias() += alpha * ma * mb;
      else
        mc.noalias() += ma * mb;
      return;
    } else {
      matrixdd_row mb((NTYPE *)B, K, N);
      if (alpha != 1)
        mc.noalias() += alpha * ma * mb;
      else
        mc.noalias() += ma * mb;
      return;
    }
  } else {
    matrixdd_row ma((NTYPE *)A, M, K);
    if (transB) {
      matrixdd_col mb((NTYPE *)B, K, N);
      if (alpha != 1)
        mc.noalias() += alpha * ma * mb;
      else
        mc.noalias() += ma * mb;
      return;
    } else {
      matrixdd_row mb((NTYPE *)B, K, N);
      if (alpha != 1)
        mc.noalias() += alpha * ma * mb;
      else
        mc.noalias() += ma * mb;
      return;
    }
  }
#endif
  throw std::invalid_argument("Not implemented for adjointd matrices (Gemm<T>).");
}

}; // namespace onnx_c_ops
