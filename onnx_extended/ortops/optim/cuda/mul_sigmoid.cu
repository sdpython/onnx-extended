#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "mul_sigmoid.h"
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

template <typename T> __device__ __inline__ T _exp_typed(const T x);

template <> __device__ __inline__ float _exp_typed(const float x) { return expf(x); }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half _exp_typed(const half x) {
  return __float2half(expf(__half2float(x)));
}
#else
template <> __device__ __inline__ half _exp_typed(const half x) { return hexp(x); }
#endif

template <typename T> __device__ __inline__ T sigmoid(const T a) {
  return a > T(0) ? (T)1 / ((T)1. + _exp_typed<T>(-a))
                  : (T)1 - (T)1 / ((T)1 + _exp_typed<T>(a));
}

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half sigmoid(const half a) {
  return __float2half(sigmoid(__half2float(a)));
}
#endif

template <typename T> __device__ __inline__ T mul_sigmoid(const T a) { return a * sigmoid(a); }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half mul_sigmoid(const half a) {
  float x = __half2float(a);
  return __float2half(x * sigmoid(x));
}
#endif

template <typename T>
__global__ void _MulSigmoidKernel(T *output_data, const T *input_data, CUDA_LONG N) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = mul_sigmoid(input_data[id]);
}

template <typename T>
void MulSigmoidImpl(cudaStream_t stream, T *output_data, const T *input_data, size_t count) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
  const int num_elements_per_thread = (N + num_threads_per_block - 1) / num_threads_per_block;

  _MulSigmoidKernel<T><<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(
      output_data, input_data, N);
}

//////////////////
// MulSigmoidOp...
//////////////////

template <typename T>
void *MulSigmoidOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<MulSigmoidKernel<T>>(api, info).release();
}

template <typename T> const char *MulSigmoidOp<T>::GetName() const { return "MulSigmoid"; }

template <typename T> const char *MulSigmoidOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t MulSigmoidOp<T>::GetInputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType MulSigmoidOp<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
ONNXTensorElementDataType MulSigmoidOp<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
MulSigmoidOp<T>::GetInputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

template <typename T> size_t MulSigmoidOp<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
MulSigmoidOp<T>::GetOutputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// MulSigmoidKernel
///////////////////

template <typename T>
MulSigmoidKernel<T>::MulSigmoidKernel(const OrtApi &api, const OrtKernelInfo *info) {}

template <typename T> void MulSigmoidKernel<T>::Compute(OrtKernelContext *context) {
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
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  output = ctx.GetOutput(0, dimsA);

  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsA));
  MulSigmoidImpl(cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
                 input_size);
}

static MulSigmoidOp<float> _kernel_f32;
static MulSigmoidOp<half> _kernel_f16;

} // namespace ortops
