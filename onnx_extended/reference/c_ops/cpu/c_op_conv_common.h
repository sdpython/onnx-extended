#include "c_op_common.h"
#include <Eigen/Dense>

using namespace Eigen;

namespace onnx_c_ops {

// The function adds value to C, assuming this array
// was initialized.
template <typename NTYPE>
void gemm(bool transA, bool transB, size_t M, size_t N, size_t K, NTYPE alpha,
          const NTYPE *A, const NTYPE *B, NTYPE beta, NTYPE *C) {

  typedef Map<Matrix<NTYPE, Dynamic, Dynamic>> matrixdd;
  if (transA) {
    if (transB) {
      matrixdd ma((NTYPE*)A, M, K);
      matrixdd mb((NTYPE*)B, K, N);
      matrixdd mc(C, M, N);
      if (beta != 1)
        mc *= beta;
      mc += ma.transpose() * ma.transpose() * alpha;
      return;
    }
    else {
      matrixdd ma((NTYPE*)A, M, K);
      matrixdd mb((NTYPE*)B, K, N);
      matrixdd mc(C, M, N);
      if (beta != 1)
        mc *= beta;
      mc += ma.transpose() * mb * alpha;
      return;
    }
  }
  else if(transB) {
      matrixdd ma((NTYPE*)A, M, K);
      matrixdd mb((NTYPE*)B, K, N);
      matrixdd mc(C, M, N);
      if (beta != 1)
        mc *= beta;
      mc += ma * mb.transpose() * alpha;
      return;
  }
  else {
      matrixdd ma((NTYPE*)A, M, K);
      matrixdd mb((NTYPE*)B, K, N);
      matrixdd mc(C, M, N);
      if (beta != 1)
        mc *= beta;
      mc += ma * mb * alpha;
      return;
  }

  if (transA) {
    if (transB) {
    } else {
      // a A B + b C, dimension = M * N
      NTYPE *begin;
      NTYPE val;
      NTYPE val0;
      size_t i, j, k, maxc = 0;
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
          maxc = maxc > (size_t)(begin - C) ? maxc : (size_t)(begin - C);
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
      size_t i, j, k, maxc = 0;
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
          maxc = maxc > (size_t)(begin - C) ? maxc : (size_t)(begin - C);
          if (maxc > M * N)
            throw std::invalid_argument("gemm00: maxc > M * N");
        }
      }
      return;
    }
  }
  throw std::invalid_argument(
      "Not implemented for transposed matrices (Gemm<T>).");
}

}; // namespace onnx_c_ops
