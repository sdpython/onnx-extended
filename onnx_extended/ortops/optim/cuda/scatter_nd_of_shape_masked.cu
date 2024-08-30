#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "scatter_nd_of_shape_masked.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define ENABLE_NCONT

namespace ortops {

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

template <typename T> __device__ __forceinline__ void _add_inplace(T &x, const T a) { x += a; }

template <> __device__ __forceinline__ void _add_inplace(half &x, const half a) {
#if __CUDA_ARCH__ < 700
  x = __float2half(__half2float(x) + __half2float(a));
#else
  x += a;
#endif
}

template <typename T>
__global__ void masked_addition_inplace_kernel(T *__restrict__ output_data,
                                               const int64_t *__restrict__ indices_data,
                                               const T *__restrict__ updates_data,
                                               const CUDA_LONG indice_size,
                                               const CUDA_LONG nrows, const CUDA_LONG stride,
                                               const int64_t masked_value) {
  auto id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= stride)
    return;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  for (size_t i = 0; i < indice_size; ++i) {
    if (indices_data[i] == masked_value)
      continue;
    _add_inplace(output_data[indices_data[i] * stride + id], updates_data[i * stride + id]);
  }
}

template <typename T, int NTHREAD>
__global__ void masked_addition_inplace_kernelN(T *__restrict__ output_data,
                                                const int64_t *__restrict__ indices_data,
                                                const T *__restrict__ updates_data,
                                                const CUDA_LONG indice_size,
                                                const CUDA_LONG nrows, const CUDA_LONG stride,
                                                const int64_t masked_value) {
  __shared__ int64_t shared_indices[NTHREAD];

  CUDA_LONG tid = threadIdx.x;
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  int begin = 0;
  int end = std::min(begin + NTHREAD, indice_size);
  while (begin < end && (end == begin + NTHREAD)) {
    shared_indices[tid] = indices_data[tid + begin];
    __syncthreads();

    for (size_t i = begin; i < end; ++i) {
      if (shared_indices[tid] == masked_value)
        continue;
      _add_inplace(output_data[shared_indices[tid] * stride + id],
                   updates_data[i * stride + id]);
    }

    begin = end;
    end = std::min(begin + NTHREAD, indice_size);
  }

  for (size_t i = begin; i < indice_size; ++i) {
    if (indices_data[i] == masked_value)
      continue;
    _add_inplace(output_data[indices_data[i] * stride + id], updates_data[i * stride + id]);
  }
}

//////////////////
// MaskedScatterNDOfShapeOp...
//////////////////

template <typename T>
void *MaskedScatterNDOfShapeOp<T>::CreateKernel(const OrtApi &api,
                                                const OrtKernelInfo *info) const {
  return std::make_unique<MaskedScatterNDOfShapeKernel<T>>(api, info).release();
}

template <typename T> const char *MaskedScatterNDOfShapeOp<T>::GetName() const {
  return "MaskedScatterNDOfShape";
}

template <typename T>
const char *MaskedScatterNDOfShapeOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t MaskedScatterNDOfShapeOp<T>::GetInputTypeCount() const {
  return 3;
};

template <>
ONNXTensorElementDataType
MaskedScatterNDOfShapeOp<float>::GetInputType(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case 2:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <>
ONNXTensorElementDataType
MaskedScatterNDOfShapeOp<half>::GetInputType(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case 2:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <typename T>
OrtMemType MaskedScatterNDOfShapeOp<T>::GetInputMemoryType(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtMemTypeCPUInput;
  case 1:
  case 2:
    return OrtMemTypeDefault;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
MaskedScatterNDOfShapeOp<T>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T> size_t MaskedScatterNDOfShapeOp<T>::GetOutputTypeCount() const {
  return 1;
}

template <>
ONNXTensorElementDataType
MaskedScatterNDOfShapeOp<float>::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <>
ONNXTensorElementDataType
MaskedScatterNDOfShapeOp<half>::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
MaskedScatterNDOfShapeOp<T>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// MaskedScatterNDOfShapeKernel
///////////////////

template <typename T>
MaskedScatterNDOfShapeKernel<T>::MaskedScatterNDOfShapeKernel(const OrtApi &api,
                                                              const OrtKernelInfo *info) {
  char value_string[1000];
  std::size_t size = 1000;
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, "reduction", value_string, &size));
  std::string value = value_string;
  if (value == "add")
    reduction_ = Reduction::Add;
  else
    EXT_THROW("unexpected reduction '", value, "'.");

  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "maskedValue", &masked_value_));

  cudaDeviceProp prop;
  int deviceId = 0;
  cudaGetDeviceProperties(&prop, deviceId);
  maxThreadPerBlock_ = prop.maxThreadsPerBlock;
}

template <typename T> void MaskedScatterNDOfShapeKernel<T>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 3, "Expected 3 inputs not ", n_inputs, ".");
  Ort::ConstValue shape = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  Ort::ConstValue updates = ctx.GetInput(2);
  Ort::UnownedValue output;

  std::vector<int64_t> dimensions = shape.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> indices_shape = indices.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> updates_shape = updates.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions.size() == 1, "shape must be a 1-dimension tensor.");

  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();

  auto memi = updates.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "updates are not on GPU");

  auto mem = shape.GetTensorMemoryInfo();
  EXT_ENFORCE(
      mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
      "The shape should be on CPU already, but mem.GetDeviceType()=", mem.GetDeviceType(), ".");
  const int64_t *X = shape.GetTensorData<int64_t>();
  std::vector<int64_t> dims(X, X + dimensions[0]);
  output = ctx.GetOutput(0, dims);

  std::vector<int64_t> input_shape = output.GetTensorTypeAndShapeInfo().GetShape();

  if (reduction_ == Reduction::Add && indices_shape[indices_shape.size() - 1] == 1 &&
      input_shape.size() == 2) {

    size_t indice_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(indices_shape));
    size_t update_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(updates_shape));

    EXT_ENFORCE(update_size == indice_size * input_shape[input_shape.size() - 1],
                "Size mismatch, update_size=", update_size, "indice_size=", indice_size,
                "input_shape[-1]=", input_shape[input_shape.size() - 1], ".");

    ComputeOptimize(stream, input_shape, indices_shape, output.GetTensorMutableData<T>(),
                    indices.GetTensorData<int64_t>(), updates.GetTensorData<T>());
  } else {
    EXT_THROW("Only add reduction and 2D tensors are supported, reduction is ", (int)reduction_,
              "input_shape.size()=", static_cast<int64_t>(input_shape.size()),
              " indices_shape[indices_shape.size() - 1]=",
              static_cast<int64_t>(indices_shape[indices_shape.size() - 1]), ".");
  }
}

template <typename T>
void _ComputeOptimize(cudaStream_t stream, const std::vector<int64_t> &input_shape,
                      const std::vector<int64_t> &indices_shape, T *output_data,
                      const int64_t *indices_data, const T *updates_data,
                      int maxThreadPerBlock_, int64_t masked_value_) {

  // The kernel is slow if there are a lot of duplicates.
  // reduction_ == Reduction::add
  // indices_shape[indices_shape.size() - 1] == 1
  // input_shape.size() == 2
  size_t indice_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(indices_shape));
  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(input_shape));
  size_t stride = input_shape[input_shape.size() - 1];
  size_t nrows = input_size / stride;

  std::vector<size_t> next_batch(indice_size);
  std::vector<uint8_t> processed(input_shape[0], 0);
  std::vector<uint8_t> processed_once(input_shape[0], 0);

#if __CUDA_ARCH__ < 700
  int threads_per_block = std::min(256, maxThreadPerBlock_ / 8);
  bool split = stride / threads_per_block <= 32;
#else
  int threads_per_block = std::min(256, maxThreadPerBlock_ / 8);
  bool split = true; // stride / threads_per_block <= 32;
#endif

  int blocks_per_grid = (stride + threads_per_block - 1) / threads_per_block;
  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);

  if (split && stride >= 256 && threads_per_block == 256) {
    masked_addition_inplace_kernelN<T, 256><<<blocks, threads, 0, stream>>>(
        output_data, indices_data, updates_data, indice_size, nrows, stride, masked_value_);
  } else if (split && stride >= 128 && threads_per_block == 128) {
    masked_addition_inplace_kernelN<T, 128><<<blocks, threads, 0, stream>>>(
        output_data, indices_data, updates_data, indice_size, nrows, stride, masked_value_);
  } else {
    masked_addition_inplace_kernel<T><<<blocks, threads, 0, stream>>>(
        output_data, indices_data, updates_data, indice_size, nrows, stride, masked_value_);
  }
}

template <typename T>
void MaskedScatterNDOfShapeKernel<T>::ComputeOptimize(cudaStream_t &stream,
                                                      const std::vector<int64_t> &input_shape,
                                                      const std::vector<int64_t> &indices_shape,
                                                      T *output_data,
                                                      const int64_t *indices_data,
                                                      const T *updates_data) const {
  _ComputeOptimize(stream, input_shape, indices_shape, output_data, indices_data, updates_data,
                   maxThreadPerBlock_, masked_value_);
}

static MaskedScatterNDOfShapeOp<float> _op32;
static MaskedScatterNDOfShapeOp<half> _op16;

} // namespace ortops
