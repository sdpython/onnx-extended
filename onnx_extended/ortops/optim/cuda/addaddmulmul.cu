#include "addaddmulmul.h"
#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include <chrono>
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
                                      const T *pB, const T *pC, size_t count,
                                      const TFunc func) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid =
      static_cast<int>(CeilDiv(count, num_threads_per_block * num_elements_per_thread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  _BinaryElementWiseSimple<T, TFunc, num_threads_per_block, num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(output_data, pA, pB, pC, N, func);
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

  EXT_ENFORCE(sizeA == sizeB && sizeB == sizeC, "The kernel does not support broadcast.");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  output = ctx.GetOutput(0, dimsA);

  if (addition) {
    BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                     A.GetTensorData<T>(), B.GetTensorData<T>(),
                                     C.GetTensorData<T>(), sizeA, Add3Op<T>());
  } else {
    BinaryElementWiseNoBroadcastImpl(cuda_stream, output.GetTensorMutableData<T>(),
                                     A.GetTensorData<T>(), B.GetTensorData<T>(),
                                     C.GetTensorData<T>(), sizeA, Mul3Op<T>());
  }
}

static AddAddMulMulOp<float, true> _add32;
static AddAddMulMulOp<half, true> _add16;
static AddAddMulMulOp<float, false> _mul32;
static AddAddMulMulOp<half, false> _mul16;

} // namespace ortops
