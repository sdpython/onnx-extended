#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "negxplus1.h"
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

template <typename T> __device__ __inline__ T _neg1plusx(const T x) {
  return (T)1 - x;
}

template <> __device__ __inline__ half _neg1plusx(const half x) {
#if __CUDA_ARCH__ < 700
  return __float2half(1 - __half2float(x));
#else
  return (half)1 - x;
#endif
}

template <typename T>
__global__ void _NegXplus1Kernel(T *output_data, const T *input_data, CUDA_LONG N) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = _neg1plusx(input_data[id]);
}

template <typename T>
void NegXplus1Impl(cudaStream_t stream, T *output_data, const T *input_data, size_t count) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  const int n_threads = GridDim::maxThreadsPerBlock;
  const int n_blocks = (N + n_threads - 1) / n_threads;

  _NegXplus1Kernel<T><<<n_blocks, n_threads, 0, stream>>>(output_data, input_data, N);
}

//////////////////
// NegXplus1Op...
//////////////////

template <typename T>
void *NegXplus1Op<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<NegXplus1Kernel<T>>(api, info).release();
}

template <typename T> const char *NegXplus1Op<T>::GetName() const { return "NegXplus1"; }

template <typename T> const char *NegXplus1Op<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t NegXplus1Op<T>::GetInputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType NegXplus1Op<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
ONNXTensorElementDataType NegXplus1Op<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
NegXplus1Op<T>::GetInputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

template <typename T> size_t NegXplus1Op<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
NegXplus1Op<T>::GetOutputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// NegXplus1Kernel
///////////////////

template <typename T>
NegXplus1Kernel<T>::NegXplus1Kernel(const OrtApi &api, const OrtKernelInfo *info) {}

template <typename T> void NegXplus1Kernel<T>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 1, "Expected 1 input not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  auto memi = A.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "first input is not on GPU");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();

  output = ctx.GetOutput(0, dimsA);

  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsA));
  NegXplus1Impl(cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(), input_size);
}

static NegXplus1Op<float> _kernel_f32;
static NegXplus1Op<int32_t> _kernel_i32;
static NegXplus1Op<half> _kernel_f16;

} // namespace ortops
