#include "addaddmulmul.h"
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

__device__ __forceinline__ void _add3_op(float *address, const float a, const float b,
                                         const float c) {
  *address = a + b + c;
}

__device__ __forceinline__ void _add3_op(half *address, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) + __half2float(b) + __half2float(c));
#else
  *address = a + b + c;
#endif
}

__device__ __forceinline__ void _mul3_op(float *address, const float a, const float b,
                                         const float c) {
  *address = a * b * c;
}

__device__ __forceinline__ void _mul3_op(half *address, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) * __half2float(b) * __half2float(c));
#else
  *address = a * b * c;
#endif
}

template <typename T> struct Mul3Op {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _mul3_op(address, a, b, c);
  }
};

template <typename T> struct Add3Op {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c) const {
    _add3_op(address, a, b, c);
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

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(T *output_data, const T *pA, const T *pB, const T *pC,
                                         CUDA_LONG N, const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_data + id, pA[id], pB[id], pC[id]);
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

template <typename T, typename TFunc>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, T *output_data, const T *pA,
                                      const T *pB, const T *pC, int64_t max_count,
                                      const TFunc func) {
  if (max_count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid =
      static_cast<int>(CeilDiv(max_count, num_threads_per_block * num_elements_per_thread));

  _BinaryElementWiseSimple<T, TFunc, num_threads_per_block, num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
          output_data, pA, pB, pC, static_cast<CUDA_LONG>(max_count), func);
}

//////////////////
// AddAddMulMulOp...
//////////////////

template <typename T, bool addition>
void *AddAddMulMulOp<T, addition>::CreateKernel(const OrtApi &api,
                                                const OrtKernelInfo *info) const {
  return std::make_unique<AddAddMulMulKernel<T, addition>>(api, info).release();
}

template <typename T, bool addition> const char *AddAddMulMulOp<T, addition>::GetName() const {
  return addition ? "AddAdd" : "MulMul";
}

template <typename T, bool addition>
const char *AddAddMulMulOp<T, addition>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T, bool addition>
size_t AddAddMulMulOp<T, addition>::GetInputTypeCount() const {
  return 3;
};

template <typename T, bool addition>
ONNXTensorElementDataType
AddAddMulMulOp<T, addition>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
ONNXTensorElementDataType
AddAddMulMulOp<T, addition>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddAddMulMulOp<T, addition>::GetInputCharacteristic(std::size_t index) const {
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
size_t AddAddMulMulOp<T, addition>::GetOutputTypeCount() const {
  return 1;
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddAddMulMulOp<T, addition>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// AddAddMulMulKernel
///////////////////

template <typename T, bool addition>
AddAddMulMulKernel<T, addition>::AddAddMulMulKernel(const OrtApi &api,
                                                    const OrtKernelInfo *info) {}

template <typename T, bool addition>
void AddAddMulMulKernel<T, addition>::Compute(OrtKernelContext *context) {
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

  if (sizeA == sizeB && sizeB == sizeC) {
    // no broadcast
    if (addition) {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), max_size, Add3Op<T>());
    } else {
      BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                       A.GetTensorData<T>(), B.GetTensorData<T>(),
                                       C.GetTensorData<T>(), max_size, Mul3Op<T>());
    }
  } else if (addition) {
    BinaryElementWiseNoBroadcastImpl(
        cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
        B.GetTensorData<T>(), C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size, Add3Op<T>());
  } else {
    BinaryElementWiseNoBroadcastImpl(
        cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
        B.GetTensorData<T>(), C.GetTensorData<T>(), sizeA, sizeB, sizeC, max_size, Mul3Op<T>());
  }
}

static AddAddMulMulOp<float, true> _add332;
static AddAddMulMulOp<half, true> _add316;
static AddAddMulMulOp<float, false> _mul332;
static AddAddMulMulOp<half, false> _mul316;

} // namespace ortops
