#include "c_op_common.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace onnx_c_ops {

namespace py = pybind11;


// The function adds value to C, assuming this array
// was initialized.
template <typename NTYPE>
void gemm(bool transA, bool transB, size_t M, size_t N, size_t K, NTYPE alpha,
          const NTYPE *A, const NTYPE *B, NTYPE beta, NTYPE *C) {
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
