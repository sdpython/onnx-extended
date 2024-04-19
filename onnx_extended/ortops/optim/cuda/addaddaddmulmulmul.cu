#include "addaddaddmulmulmul.h"
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

__device__ __forceinline__ void _add4_op(float *address, const float a, const float b,
                                         const float c, const float d) {
  *address = a + b + c + d;
}

__device__ __forceinline__ void _add4_op(half *address, const half a, const half b,
                                         const half c, const half d) {
#if __CUDA_ARCH__ < 700
  *address =
      __float2half(__half2float(a) + __half2float(b) + __half2float(c) + __half2float(d));
#else
  *address = a + b + c + d;
#endif
}

__device__ __forceinline__ void _mul4_op(float *address, const float a, const float b,
                                         const float c, const float d) {
  *address = a * b * c * d;
}

__device__ __forceinline__ void _mul4_op(half *address, const half a, const half b,
                                         const half c, const half d) {
#if __CUDA_ARCH__ < 700
  *address =
      __float2half(__half2float(a) * __half2float(b) * __half2float(c) * __half2float(d));
#else
  *address = a * b * c * d;
#endif
}

template <typename T> struct Mul4Op {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c,
                                        const T d) const {
    _mul4_op(address, a, b, c, d);
  }
};

template <typename T> struct Add4Op {
  __device__ __inline__ void operator()(T *address, const T a, const T b, const T c,
                                        const T d) const {
    _add4_op(address, a, b, c, d);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(T *output_data, const T *pA, const T *pB, const T *pC,
                                         const T *pD, CUDA_LONG N, const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_data + id, pA[id], pB[id], pC[id], pD[id]);
      id += NumThreadsPerBlock;
    }
  }
}

template <class INT, class INT2> inline __host__ __device__ INT CeilDiv(INT a, INT2 b) {
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);
}

template <typename T, typename TFunc>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, T *output_data, const T *pA,
                                      const T *pB, const T *pC, const T *pD, size_t count,
                                      const TFunc func) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  const int num_elements_per_thread = GridDim::maxElementsPerThread;
  const int num_threads_per_block = GridDim::maxThreadsPerBlock;

  int blocksPerGrid =
      static_cast<int>(CeilDiv(count, num_threads_per_block * num_elements_per_thread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  _BinaryElementWiseSimple<T, TFunc, num_threads_per_block, num_elements_per_thread>
      <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(output_data, pA, pB, pC, pD, N,
                                                            func);
}

//////////////////
// AddAddAddMulMulMulOp...
//////////////////

template <typename T, bool addition>
void *AddAddAddMulMulMulOp<T, addition>::CreateKernel(const OrtApi &api,
                                                      const OrtKernelInfo *info) const {
  return std::make_unique<AddAddAddMulMulMulKernel<T, addition>>(api, info).release();
}

template <typename T, bool addition>
const char *AddAddAddMulMulMulOp<T, addition>::GetName() const {
  return addition ? "AddAddAdd" : "MulMulMul";
}

template <typename T, bool addition>
const char *AddAddAddMulMulMulOp<T, addition>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T, bool addition>
size_t AddAddAddMulMulMulOp<T, addition>::GetInputTypeCount() const {
  return 4;
};

template <typename T, bool addition>
ONNXTensorElementDataType
AddAddAddMulMulMulOp<T, addition>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
ONNXTensorElementDataType
AddAddAddMulMulMulOp<T, addition>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddAddAddMulMulMulOp<T, addition>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
  case 3:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T, bool addition>
size_t AddAddAddMulMulMulOp<T, addition>::GetOutputTypeCount() const {
  return 1;
}

template <typename T, bool addition>
OrtCustomOpInputOutputCharacteristic
AddAddAddMulMulMulOp<T, addition>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// AddAddAddMulMulMulKernel
///////////////////

template <typename T, bool addition>
AddAddAddMulMulMulKernel<T, addition>::AddAddAddMulMulMulKernel(const OrtApi &api,
                                                                const OrtKernelInfo *info) {}

template <typename T, bool addition>
void AddAddAddMulMulMulKernel<T, addition>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 4, "Expected 3 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue B = ctx.GetInput(1);
  Ort::ConstValue C = ctx.GetInput(2);
  Ort::ConstValue D = ctx.GetInput(3);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsB = B.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsC = C.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dimsD = D.GetTensorTypeAndShapeInfo().GetShape();

  auto sizeA = onnx_c_ops::flattened_dimension(dimsA);
  auto sizeB = onnx_c_ops::flattened_dimension(dimsB);
  auto sizeC = onnx_c_ops::flattened_dimension(dimsC);
  auto sizeD = onnx_c_ops::flattened_dimension(dimsD);

  EXT_ENFORCE(sizeA == sizeB && sizeB == sizeC && sizeC == sizeD,
              "The kernel does not support broadcast.");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  output = ctx.GetOutput(0, dimsA);

  if (addition) {
    BinaryElementWiseNoBroadcastImpl(
        cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
        B.GetTensorData<T>(), C.GetTensorData<T>(), D.GetTensorData<T>(), sizeA, Add4Op<T>());
  } else {
    BinaryElementWiseNoBroadcastImpl(
        cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
        B.GetTensorData<T>(), C.GetTensorData<T>(), D.GetTensorData<T>(), sizeA, Mul4Op<T>());
  }
}

static AddAddAddMulMulMulOp<float, true> _add432;
static AddAddAddMulMulMulOp<half, true> _add416;
static AddAddAddMulMulMulOp<float, false> _mul432;
static AddAddAddMulMulMulOp<half, false> _mul416;

} // namespace ortops
