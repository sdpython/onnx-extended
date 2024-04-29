#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "submul.h"
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

__device__ __forceinline__ void _submul_op(float *address, const float a, const float b,
                                           const float c) {
  *address = (a - b) * c;
}

__device__ __forceinline__ void _submul_op(half *address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half((__half2float(a) - __half2float(b)) * __half2float(c));
#else
  *address = (a - b) * c;
#endif
}

__device__ __forceinline__ void _submul_neg_op(float *address, const float a, const float b,
                                               const float c) {
  *address = (b - a) * c;
}

__device__ __forceinline__ void _submul_neg_op(half *address, const half a, const half b,
                                               const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half((__half2float(b) - __half2float(a)) * __half2float(c));
#else
  *address = (b - a) * c;
#endif
}

__device__ __forceinline__ void _mulsub_op(float *address, const float a, const float b,
                                           const float c) {
  *address = a * b - c;
}

__device__ __forceinline__ void _mulsub_op(half *address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) * __half2float(b) - __half2float(c));
#else
  *address = a * b - c;
#endif
}

__device__ __forceinline__ void _mulsub_neg_op(float *address, const float a, const float b,
                                               const float c) {
  *address = c - a * b;
}

__device__ __forceinline__ void _mulsub_neg_op(half *address, const half a, const half b,
                                               const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(c) - __half2float(a) * __half2float(b));
#else
  *address = c - a * b;
#endif
}

template <typename T> struct SubMul {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _submul_op(address, a, b, c);
  }
};

template <typename T> struct MulSub {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _mulsub_op(address, a, b, c);
  }
};

template <typename T> struct SubMulNeg {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _submul_neg_op(address, a, b, c);
  }
};

template <typename T> struct MulSubNeg {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _mulsub_neg_op(address, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(T *output_data, const T *pA, const T *pB, const T *pC,
                                         CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC, CUDA_LONG N,
                                         const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_data + id, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <class INT, class INT2> inline __host__ __device__ INT CeilDiv(INT a, INT2 b) {
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);
}

template <typename T, typename TFunc>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, T *output_data, const T *pA,
                                      const T *pB, const T *pC, int64_t countA, int64_t countB,
                                      int64_t countC, int64_t max_count, const TFunc func) {
  if (max_count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid =
      static_cast<int>(CeilDiv(max_count, num_threads_per_block * num_elements_per_thread));

  _BinaryElementWiseSimple<T, TFunc, num_threads_per_block, num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
          output_data, pA, pB, pC, static_cast<CUDA_LONG>(countA),
          static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
          static_cast<CUDA_LONG>(max_count), func);
}

//////////////////
// SubMulOp...
//////////////////

template <typename T, bool addition>
void *SubMulOp<T, addition>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<SubMulKernel<T, addition>>(api, info).release();
}

template <typename T, bool addition> const char *SubMulOp<T, addition>::GetName() const {
  return addition ? "SubMul" : "MulSub";
}

template <typename T, bool addition>
const char *SubMulOp<T, addition>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T, bool addition> size_t SubMulOp<T, addition>::GetInputTypeCount() const {
  return 3;
};

template <typename T, bool addition>
ONNXTensorElementDataType SubMulOp<T, addition>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
ONNXTensorElementDataType SubMulOp<T, addition>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
SubMulOp<T, addition>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T, bool addition> size_t SubMulOp<T, addition>::GetOutputTypeCount() const {
  return 1;
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
SubMulOp<T, addition>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// SubMulKernel
///////////////////

template <typename T, bool addition>
SubMulKernel<T, addition>::SubMulKernel(const OrtApi &api, const OrtKernelInfo *info) {
  negative_ = KernelInfoGetOptionalAttributeInt64AsBool(api, info, "negative", false);
}

template <typename T, bool addition>
void SubMulKernel<T, addition>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 3, "Expected 3 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue B = ctx.GetInput(1);
  Ort::ConstValue C = ctx.GetInput(2);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsB = B.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsC = C.GetTensorTypeAndShapeInfo().GetShape();

  auto sizeA = onnx_c_ops::flattened_dimension(dimsA);
  auto sizeB = onnx_c_ops::flattened_dimension(dimsB);
  auto sizeC = onnx_c_ops::flattened_dimension(dimsC);
  auto max_size = std::max(sizeA, std::max(sizeB, sizeC));

  auto max_rank = std::max(dimsA.size(), std::max(dimsB.size(), dimsC.size()));
  while (dimsA.size() < max_rank)
    dimsA.insert(dimsA.begin(), 1);
  while (dimsB.size() < max_rank)
    dimsB.insert(dimsB.begin(), 1);
  while (dimsC.size() < max_rank)
    dimsC.insert(dimsC.begin(), 1);

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  std::vector<int64_t> output_dims(dimsA.size());
  for (size_t i = 0; i < dimsA.size(); ++i) {
    output_dims[i] = std::max(std::max(dimsA[i], dimsB[i]), dimsC[i]);
  }
  output = ctx.GetOutput(0, output_dims);

  if (addition) {
    if (negative_) {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size,
                                       SubMulNeg<T>());
    } else {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size,
                                       SubMul<T>());
    }
  } else {
    if (negative_) {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size,
                                       MulSubNeg<T>());
    } else {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size,
                                       MulSub<T>());
    }
  }
}

static SubMulOp<float, true> _submul32;
static SubMulOp<half, true> _submul16;
static SubMulOp<float, false> _mulsub32;
static SubMulOp<half, false> _mulsub16;

} // namespace ortops
