#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "replace_zero.h"
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

template <typename T> __device__ __inline__ T _replace_zero(const T x, const T by) {
  return x == (T)0 ? by : x;
}

template <> __device__ __inline__ half _replace_zero(const half x, const half by) {
#if __CUDA_ARCH__ < 700
  return __half2float(x) == 0 ? by : x;
#else
  return x == (half)0 ? by : x;
#endif
}

template <typename T>
__global__ void _ReplaceZeroKernel(T *output_data, const T *input_data, CUDA_LONG N, const T by) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = _replace_zero(input_data[id], by);
}

template <typename T> T _cvt(float value) { return (T)value; }

template <> half _cvt(float value) { return __float2half(value); }

template <typename T>
void ReplaceZeroImpl(cudaStream_t stream, T *output_data, const T *input_data, size_t count,
                     float by) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
  const int num_elements_per_thread = (N + num_threads_per_block - 1) / num_threads_per_block;

  T cby = _cvt<T>(by);

  _ReplaceZeroKernel<T><<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(
      output_data, input_data, N, cby);
}

//////////////////
// ReplaceZeroOp...
//////////////////

template <typename T>
void *ReplaceZeroOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<ReplaceZeroKernel<T>>(api, info).release();
}

template <typename T> const char *ReplaceZeroOp<T>::GetName() const { return "ReplaceZero"; }

template <typename T> const char *ReplaceZeroOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t ReplaceZeroOp<T>::GetInputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType ReplaceZeroOp<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
ONNXTensorElementDataType ReplaceZeroOp<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
ReplaceZeroOp<T>::GetInputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

template <typename T> size_t ReplaceZeroOp<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
ReplaceZeroOp<T>::GetOutputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// ReplaceZeroKernel
///////////////////

template <typename T>
ReplaceZeroKernel<T>::ReplaceZeroKernel(const OrtApi &api, const OrtKernelInfo *info) {
  ThrowOnError(api, api.KernelInfoGetAttribute_float(info, "by", &by_));
}

template <typename T> void ReplaceZeroKernel<T>::Compute(OrtKernelContext *context) {
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
  ReplaceZeroImpl(cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
                  input_size, by_);
}

static ReplaceZeroOp<float> _kernel_f32;
static ReplaceZeroOp<half> _kernel_f16;

} // namespace ortops
