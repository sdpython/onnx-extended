#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "mul_mul_sigmoid.h"
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

template <typename T> __device__ __inline__ T mul_mul_sigmoid(const T x, const T y) {
  return x * y * sigmoid(y);
}

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half mul_mul_sigmoid(const half x, const half y) {
  float hy = __half2float(y);
  return __float2half(__half2float(x) * hy * sigmoid(hy));
}
#endif

template <typename T>
__global__ void _MulMulSigmoidKernel(T *output_data, const T *px, const T *py, CUDA_LONG N,
                                     CUDA_LONG Nx, CUDA_LONG Ny) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = mul_mul_sigmoid(px[id % Nx], py[id % Ny]);
}

template <typename T>
void MulMulSigmoidImpl(cudaStream_t stream, T *output_data, const T *px, const T *py,
                       size_t count_x, size_t count_y) {
  if (count_x == 0 || count_y == 0)
     // special case where there's a dim value of 0 in the output shape
    return;

  CUDA_LONG N = static_cast<CUDA_LONG>(std::max(count_x, count_y));

  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
  const int num_elements_per_thread = (N + num_threads_per_block - 1) / num_threads_per_block;

  _MulMulSigmoidKernel<T><<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(
      output_data, px, py, N, static_cast<CUDA_LONG>(count_x), static_cast<CUDA_LONG>(count_y));
}

//////////////////
// MulMulSigmoidOp...
//////////////////

template <typename T>
void *MulMulSigmoidOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<MulMulSigmoidKernel<T>>(api, info).release();
}

template <typename T> const char *MulMulSigmoidOp<T>::GetName() const {
  return "MulMulSigmoid";
}

template <typename T> const char *MulMulSigmoidOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t MulMulSigmoidOp<T>::GetInputTypeCount() const { return 2; };

template <typename T>
ONNXTensorElementDataType MulMulSigmoidOp<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
ONNXTensorElementDataType MulMulSigmoidOp<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
MulMulSigmoidOp<T>::GetInputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

template <typename T> size_t MulMulSigmoidOp<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
MulMulSigmoidOp<T>::GetOutputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// MulMulSigmoidKernel
///////////////////

template <typename T>
MulMulSigmoidKernel<T>::MulMulSigmoidKernel(const OrtApi &api, const OrtKernelInfo *info) {}

template <typename T> void MulMulSigmoidKernel<T>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 2, "Expected 2 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue B = ctx.GetInput(1);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  auto memi = A.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "first input is not on GPU");

  std::vector<int64_t> dimsB = B.GetTensorTypeAndShapeInfo().GetShape();
  memi = B.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "second input is not on GPU");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));
  size_t input_size_a = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsA));
  size_t input_size_b = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsB));

  output = ctx.GetOutput(0, input_size_a < input_size_b ? dimsB : dimsA);

  MulMulSigmoidImpl(cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(),
                    B.GetTensorData<T>(), input_size_a, input_size_b);
}

static MulMulSigmoidOp<float> _kernel_f32;
static MulMulSigmoidOp<half> _kernel_f16;

} // namespace ortops
