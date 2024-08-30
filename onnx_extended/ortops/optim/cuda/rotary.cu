#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "rotary.h"
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

template <typename T> __device__ __inline__ T _neg(const T x) { return -x; }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half _neg(const half x) {
  return __float2half(-__half2float(x));
}
#endif

template <typename T, RotarySide side>
__global__ void _RotaryKernelLeft(T *output_data, const T *input_data, CUDA_LONG half_N,
                                  CUDA_LONG half_stride) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= half_N)
    return;
  CUDA_LONG last = id % half_stride;
  id = (id - last) * 2 + last;
  if (side == RotarySide::RIGHT) {
    output_data[id + half_stride] = input_data[id];
    output_data[id] = _neg(input_data[id + half_stride]);
  } else {
    output_data[id + half_stride] = _neg(input_data[id]);
    output_data[id] = input_data[id + half_stride];
  }
}

template <typename T>
void RotaryImpl(cudaStream_t stream, T *output_data, const T *input_data, size_t count,
                size_t last_dim, RotarySide side) {
  if (count == 0) // special case where there's a dim value of 0 in the output shape
    return;

  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  CUDA_LONG stride = static_cast<CUDA_LONG>(last_dim);

  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
  const int num_elements_per_thread =
      (N / 2 + num_threads_per_block - 1) / num_threads_per_block;

  switch (side) {
  case RotarySide::LEFT:
    _RotaryKernelLeft<T, RotarySide::LEFT>
        <<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(output_data, input_data,
                                                                        N / 2, stride / 2);
    break;
  case RotarySide::RIGHT:
    _RotaryKernelLeft<T, RotarySide::RIGHT>
        <<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(output_data, input_data,
                                                                        N / 2, stride / 2);
    break;
  }
}

//////////////////
// RotaryOp...
//////////////////

template <typename T>
void *RotaryOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<RotaryKernel<T>>(api, info).release();
}

template <typename T> const char *RotaryOp<T>::GetName() const { return "Rotary"; }

template <typename T> const char *RotaryOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t RotaryOp<T>::GetInputTypeCount() const { return 2; };

template <typename T>
ONNXTensorElementDataType RotaryOp<T>::GetInputType(std::size_t index) const {
  switch (index) {
  case 0:
    return CTypeToOnnxType<T>().onnx_type();
  case 1:
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  default:
    EXT_THROW("Input index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T>
ONNXTensorElementDataType RotaryOp<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
RotaryOp<T>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T> OrtMemType RotaryOp<T>::GetInputMemoryType(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtMemTypeDefault;
  case 1:
    return OrtMemTypeCPUInput;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <typename T> size_t RotaryOp<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
RotaryOp<T>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// RotaryKernel
///////////////////

template <typename T>
RotaryKernel<T>::RotaryKernel(const OrtApi &api, const OrtKernelInfo *info) {
  char value_string[1000];
  std::size_t size = 1000;
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, "side", value_string, &size));
  std::string value = value_string;
  if (value == "left")
    rotary_side_ = RotarySide::LEFT;
  else if (value == "right")
    rotary_side_ = RotarySide::RIGHT;
  else
    EXT_THROW("unexpected side '", value, "'.");
}

template <typename T> void RotaryKernel<T>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 2, "Expected 2 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue split = ctx.GetInput(1);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> dims_split = split.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dims_split.size() == 1 && dims_split[0] == 2,
              "Rotary only works when there are two sides but size=", dims_split.size(),
              " and dims_split=", dims_split.size() > 0 ? dims_split[0] : -1, ".");

  auto mem = split.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
              "The splits should be on CPU already, but mem.GetDeviceType()=",
              mem.GetDeviceType(), ".");
  const int64_t *splits = split.GetTensorData<int64_t>();
  EXT_ENFORCE(splits[0] == splits[1], "Only equal split are allowed not ", splits[0], " and ",
              splits[1], ".");
  EXT_ENFORCE(splits[0] + splits[1] == dimsA[dimsA.size() - 1], "Sum of the splits ",
              splits[0] + splits[1], " are not equal to the last dimension ",
              dimsA[dimsA.size() - 1], ".");

  auto memi = A.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "first input is not on GPU");

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  output = ctx.GetOutput(0, dimsA);

  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsA));
  RotaryImpl(cuda_stream, output.GetTensorMutableData<T>(), A.GetTensorData<T>(), input_size,
             dimsA[dimsA.size() - 1], rotary_side_);
}

static RotaryOp<float> _rot_f32;
static RotaryOp<half> _rot_f16;

} // namespace ortops
