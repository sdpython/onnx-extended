#include "add_or_mul_shared_input.h"
#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ortops {

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256, // max threads per block
    maxElementsPerThread = 4, // max element processed per thread
  };
};

__device__ __forceinline__ void _add3_op(float *ab, float *ac, const float a, const float b,
                                         const float c) {
  *ab = a + b;
  *ac = a + c;
}

__device__ __forceinline__ void _add3_op(half *ab, half *ac, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *ab = __float2half(__half2float(a) + __half2float(b));
  *ac = __float2half(__half2float(a) + __half2float(c));
#else
  *ab = a + b;
  *ac = a + c;
#endif
}

__device__ __forceinline__ void _mul3_op(float *ab, float *ac, const float a, const float b,
                                         const float c) {
  *ab = a * b;
  *ac = a * c;
}

__device__ __forceinline__ void _mul3_op(half *ab, half *ac, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *ab = __float2half(__half2float(a) * __half2float(b));
  *ac = __float2half(__half2float(a) * __half2float(c));
#else
  *ab = a * b;
  *ac = a * c;
#endif
}

template <typename T> struct Mul3SharedOp {
  __device__ __inline__ void operator()(T *ab, T *ac, const T a, const T b, const T c) const {
    _mul3_op(ab, ac, a, b, c);
  }
};

template <typename T> struct Add3SharedOp {
  __device__ __inline__ void operator()(T *ab, T *ac, const T a, const T b, const T c) const {
    _add3_op(ab, ac, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(T *output_ab, T *output_ac, const T *pA, const T *pB,
                                         const T *pC, CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC,
                                         CUDA_LONG N, const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_ab + id, output_ac + id, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <class INT, class INT2> inline __host__ __device__ INT CeilDiv(INT a, INT2 b) {
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);
}

template <typename T, typename TFunc>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, T *output_ab, T *output_ac,
                                      const T *pA, const T *pB, const T *pC, int64_t countA,
                                      int64_t countB, int64_t countC, int64_t max_count,
                                      const TFunc func) {
  if (max_count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid =
      static_cast<int>(CeilDiv(max_count, num_threads_per_block * num_elements_per_thread));

  _BinaryElementWiseSimple<T, TFunc, num_threads_per_block, num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
          output_ab, output_ac, pA, pB, pC, static_cast<CUDA_LONG>(countA),
          static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
          static_cast<CUDA_LONG>(max_count), func);
}

//////////////////
// AddOrMulSharedInputOp...
//////////////////

template <typename T, bool addition>
void *AddOrMulSharedInputOp<T, addition>::CreateKernel(const OrtApi &api,
                                                       const OrtKernelInfo *info) const {
  return std::make_unique<AddOrMulSharedInputKernel<T, addition>>(api, info).release();
}

template <typename T, bool addition>
const char *AddOrMulSharedInputOp<T, addition>::GetName() const {
  return addition ? "AddSharedInput" : "MulSharedInput";
}

template <typename T, bool addition>
const char *AddOrMulSharedInputOp<T, addition>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T, bool addition>
size_t AddOrMulSharedInputOp<T, addition>::GetInputTypeCount() const {
  return 3;
};

template <typename T, bool addition>
ONNXTensorElementDataType
AddOrMulSharedInputOp<T, addition>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
ONNXTensorElementDataType
AddOrMulSharedInputOp<T, addition>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddOrMulSharedInputOp<T, addition>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T, bool addition>
size_t AddOrMulSharedInputOp<T, addition>::GetOutputTypeCount() const {
  return 2;
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddOrMulSharedInputOp<T, addition>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// AddOrMulSharedInputKernel
///////////////////

bool _check_shape(const std::vector<int64_t> &shape) {
  bool met_non_one = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1) {
      met_non_one = true;
      break;
    }
    if (shape[i] == 1 && met_non_one) {
      return false;
    }
  }
  return true;
}

template <typename T, bool addition>
AddOrMulSharedInputKernel<T, addition>::AddOrMulSharedInputKernel(const OrtApi &api,
                                                                  const OrtKernelInfo *info) {}

template <typename T, bool addition>
void AddOrMulSharedInputKernel<T, addition>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 3, "Expected 3 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue B = ctx.GetInput(1);
  Ort::ConstValue C = ctx.GetInput(2);

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsB = B.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsC = C.GetTensorTypeAndShapeInfo().GetShape();

  int64_t sizeA = onnx_c_ops::flattened_dimension(dimsA);
  int64_t sizeB = onnx_c_ops::flattened_dimension(dimsB);
  int64_t sizeC = onnx_c_ops::flattened_dimension(dimsC);

  // Computes AB, AC.

  auto max_rank = std::max(dimsA.size(), std::max(dimsB.size(), dimsC.size()));
  while (dimsA.size() < max_rank)
    dimsA.insert(dimsA.begin(), 1);
  while (dimsB.size() < max_rank)
    dimsB.insert(dimsB.begin(), 1);
  while (dimsC.size() < max_rank)
    dimsC.insert(dimsC.begin(), 1);

  int64_t max_dim = std::max(std::max(sizeA, sizeB), sizeC);
  EXT_ENFORCE(_check_shape(dimsA), "Shape of A", dimsA, " is not supported for this operator.");
  EXT_ENFORCE(_check_shape(dimsB), "Shape of B", dimsB, " is not supported for this operator.");
  EXT_ENFORCE(_check_shape(dimsC), "Shape of C", dimsC, " is not supported for this operator.");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  std::vector<int64_t> output_dims(dimsA.size());
  for (size_t i = 0; i < dimsA.size(); ++i) {
    output_dims[i] = std::max(dimsA[i], dimsB[i]);
  }

  Ort::UnownedValue output_ab = ctx.GetOutput(0, output_dims);
  Ort::UnownedValue output_ac = ctx.GetOutput(1, output_dims);

  if (addition) {
    BinaryElementWiseNoBroadcastImpl(cuda_stream, output_ab.GetTensorMutableData<T>(),
                                     output_ac.GetTensorMutableData<T>(), A.GetTensorData<T>(),
                                     B.GetTensorData<T>(), C.GetTensorData<T>(), sizeA, sizeB,
                                     sizeC, max_dim, Add3SharedOp<T>());
  } else {
    BinaryElementWiseNoBroadcastImpl(cuda_stream, output_ab.GetTensorMutableData<T>(),
                                     output_ac.GetTensorMutableData<T>(), A.GetTensorData<T>(),
                                     B.GetTensorData<T>(), C.GetTensorData<T>(), sizeA, sizeB,
                                     sizeC, max_dim, Mul3SharedOp<T>());
  }
}

static AddOrMulSharedInputOp<float, true> _add332;
static AddOrMulSharedInputOp<half, true> _add316;
static AddOrMulSharedInputOp<float, false> _mul332;
static AddOrMulSharedInputOp<half, false> _mul316;

} // namespace ortops
