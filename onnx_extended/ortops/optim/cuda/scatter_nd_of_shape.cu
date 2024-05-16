#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "scatter_nd_of_shape.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ortops {

__device__ __forceinline__ void atomic_add(float *address, float value) {
  atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double *address, double value) {
#if __CUDA_ARCH__ < 600
  unsigned long long *raw_address = reinterpret_cast<unsigned long long *>(address);
  unsigned long long raw_old_value = 0ULL;
  unsigned long long raw_new_value = 0ULL;
  unsigned long long seen_old_value = 0ULL;
  double *const p_old_value = reinterpret_cast<double *>(&raw_old_value);
  double *const p_new_value = reinterpret_cast<double *>(&raw_new_value);
  do {
    *p_old_value = *address;
    *p_new_value = *address + value;
    seen_old_value = atomicCAS(raw_address, raw_old_value, raw_new_value);
  } while (seen_old_value != raw_old_value);
#else
  atomicAdd(address, value);
#endif
}

//
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh
//
__device__ __forceinline__ void atomic_add(half *address, half value) {
#if __CUDA_ARCH__ < 700
  unsigned int *base_address = (unsigned int *)((char *)address - ((size_t)address & 2));
  unsigned int old = *base_address;
  unsigned int assumed;
  unsigned short x;

  do {
    assumed = old;
    x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    x = __half_as_short(__float2half(__half2float(*reinterpret_cast<const __half *>(&x)) +
                                     __half2float(value)));
    old = (size_t)address & 2 ? (old & 0xffff) | (x << 16) : (old & 0xffff0000) | x;
    old = atomicCAS(base_address, assumed, old);
  } while (assumed != old);
#else
  atomicAdd(address, value);
#endif
}

template <class T> struct FuncAdd {
  __device__ __inline__ void operator()(T *start_addr, T value) const {
    atomic_add(start_addr, value);
  }
};

#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)                                             \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;                                         \
  if (id >= N)                                                                                 \
    return;

template <typename T, typename TFunc>
__global__ void
_ScatterNDKernelReduction(T *output_data, const size_t num_indices, const int64_t *indices_data,
                          const int64_t last_index_dimension,
                          Shape2 element_counts_and_input_dims, const T *updates_data,
                          const size_t num_updates_elements, const TFunc func) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims.dims[i - indices_start];
    int64_t dim_value =
        element_counts_and_input_dims.dims[i - indices_start + last_index_dimension];

    // Clamp the index if out of range
    // This would have been an error in the CPU kernel, but throwing in the CUDA EP
    // is hard. This is the approach taken by other frameworks for out of bound indices
    // in their corresponding GPU backends as well.
    // index >= -dim_value && index < dim_value

    if (index >= 0) {
      if (index >= dim_value) {
        index = dim_value - 1;
      }
    } else {
      if (index < -dim_value) {
        index = 0;
      } else {
        index += dim_value;
      }
    }

    data_offset += (index * element_count_dim);
  }

  const T *updates_data_base = updates_data + num_updates_elements * id;
  T *output_data_base = output_data + data_offset;

  for (size_t i = 0; i < num_updates_elements; ++i) {
    func(output_data_base + i, updates_data_base[i]);
  }
}

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256, // max threads per block
    maxElementsPerThread = 4, // max element processed per thread
  };
};

void ScatterNDImplReduction(cudaStream_t stream, void *output_data, const int32_t element_type,
                            const size_t num_indices, const int64_t *indices_data,
                            const int64_t last_index_dimension,
                            const Shape2 &element_counts_and_input_dims,
                            const void *updates_data, const size_t num_updates_elements,
                            Reduction reduction) {
  if (num_indices == 0)
    return;

  // Parallelize on number of indices
  int blocksPerGrid =
      static_cast<int>(ceil(static_cast<float>(num_indices) / GridDim::maxThreadsPerBlock));

  switch (reduction) {
  case Reduction::Add:
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<float *>(output_data), num_indices, indices_data,
          last_index_dimension, element_counts_and_input_dims,
          reinterpret_cast<const float *>(updates_data), num_updates_elements,
          FuncAdd<float>());
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<half *>(output_data), num_indices, indices_data,
          last_index_dimension, element_counts_and_input_dims,
          reinterpret_cast<const half *>(updates_data), num_updates_elements, FuncAdd<half>());
      break;

    default:
      EXT_THROW("Type ", element_type, " not supported for ScatterND operator.");
    }
    break;

  default:
    EXT_THROW("Reduction ", static_cast<int>(reduction),
              " not implemented for ScatterND operator.");
  }
}

struct TensorPitches : std::vector<int64_t> {
  TensorPitches(const std::vector<int64_t> &dims, size_t rank = 0)
      : std::vector<int64_t>(std::max(rank, dims.size()), 0) {
    Calculate(*this, dims);
  }

  static bool Calculate(std::vector<int64_t> &p, const std::vector<int64_t> &dims) {
    // The pitches is the size of the next inner axis. Aka the amount to move by one of the next
    // inner axis. For a tensor with shape(2,3,4,5) the values would be: (3*4*5, 4*5, 5, 1) Note
    // that the outermost '2' is never used, as you never need to move by the entire size of the
    // outermost axis

    auto tensor_rank = dims.size();
    auto pitch_rank = p.size();
    auto padded_rank = pitch_rank - tensor_rank;
    if (static_cast<ptrdiff_t>(padded_rank) < 0)
      return false;

    // Guard against Scalars
    if (pitch_rank == 0) {
      return true;
    }

    *(p.rbegin()) = 1; // The innermost axis is 1 (single values)
    if (tensor_rank > 1) {
      for (size_t i = tensor_rank - 1; i-- > 0;) {
        p[i + padded_rank] = p[i + 1 + padded_rank] * dims[i + 1];
      }
    }

    if (padded_rank >= 1) {
      for (size_t i = 0; i < padded_rank; ++i) {
        if (i == 0 && tensor_rank > 0)
          // For scalar tensor, the values in the pitches are all 1.
          p[padded_rank - 1] = p[padded_rank] * dims[0];
        else
          p[padded_rank - 1 - i] = p[padded_rank - 1];
      }
    }
    return true;
  }
};

template <typename T> __device__ __forceinline__ void _add_inplace(T &x, const T a) { x += a; }

template <> __device__ __forceinline__ void _add_inplace(half &x, const half a) {
#if __CUDA_ARCH__ < 700
  x = __float2half(__half2float(x) + __half2float(a));
#else
  x += a;
#endif
}

template <typename T>
__global__ void
addition_inplace_kernel(T *__restrict__ output_data, const int64_t *__restrict__ indices_data,
                        const T *__restrict__ updates_data, const CUDA_LONG indice_size,
                        const CUDA_LONG nrows, const CUDA_LONG stride) {
  auto id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= stride)
    return;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  int64_t index;
  for (size_t i = 0; i < indice_size; ++i) {
    index = (indices_data[i] + nrows) % nrows;
    _add_inplace(output_data[index * stride + id], updates_data[i * stride + id]);
  }
}

//////////////////
// ScatterNDOfShapeOp...
//////////////////

template <typename T>
void *ScatterNDOfShapeOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<ScatterNDOfShapeKernel<T>>(api, info).release();
}

template <typename T> const char *ScatterNDOfShapeOp<T>::GetName() const {
  return "ScatterNDOfShape";
}

template <typename T> const char *ScatterNDOfShapeOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t ScatterNDOfShapeOp<T>::GetInputTypeCount() const { return 3; };

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<float>::GetInputType(std::size_t index) const {
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
ONNXTensorElementDataType ScatterNDOfShapeOp<half>::GetInputType(std::size_t index) const {
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
OrtMemType ScatterNDOfShapeOp<T>::GetInputMemoryType(std::size_t index) const {
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
ScatterNDOfShapeOp<T>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T> size_t ScatterNDOfShapeOp<T>::GetOutputTypeCount() const { return 1; }

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<float>::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<half>::GetOutputType(std::size_t index) const {
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
ScatterNDOfShapeOp<T>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// ScatterNDOfShapeKernel
///////////////////

template <typename T>
ScatterNDOfShapeKernel<T>::ScatterNDOfShapeKernel(const OrtApi &api,
                                                  const OrtKernelInfo *info) {
  char value_string[1000];
  std::size_t size = 1000;
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, "reduction", value_string, &size));
  std::string value = value_string;
  if (value == "add")
    reduction_ = Reduction::Add;
  else
    EXT_THROW("unexpected reduction '", value, "'.");

  value = KernelInfoGetOptionalAttributeString(api, info, "strategy", "none");
  if (value == "none")
    strategy_ = Strategy::None;
  else if (value == "optimize")
    strategy_ = Strategy::Optimize;
  else
    EXT_THROW("unexpected strategy '", value, "'.");

  cudaDeviceProp prop;
  int deviceId = 0;
  cudaGetDeviceProperties(&prop, deviceId);
  maxThreadPerBlock_ = prop.maxThreadsPerBlock;
}

template <typename T> void ScatterNDOfShapeKernel<T>::Compute(OrtKernelContext *context) {
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

  if (reduction_ == Reduction::Add && strategy_ == Strategy::Optimize &&
      indices_shape[indices_shape.size() - 1] == 1 && input_shape.size() == 2 &&
      input_shape[input_shape.size() - 1] >= maxThreadPerBlock_) {

    size_t indice_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(indices_shape));
    size_t update_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(updates_shape));

    EXT_ENFORCE(update_size == indice_size * input_shape[input_shape.size() - 1],
                "Size mismatch, update_size=", update_size, "indice_size=", indice_size,
                "input_shape[-1]=", input_shape[input_shape.size() - 1], ".");

    ComputeOptimize(stream, input_shape, indices_shape, output.GetTensorMutableData<T>(),
                    indices.GetTensorData<int64_t>(), updates.GetTensorData<T>());
  } else {
    ComputeNone(stream, input_shape, indices_shape, output.GetTensorMutableData<T>(),
                indices.GetTensorData<int64_t>(), updates.GetTensorData<T>());
  }
}

template <typename T>
void ScatterNDOfShapeKernel<T>::ComputeNone(cudaStream_t &stream,
                                            const std::vector<int64_t> &input_shape,
                                            const std::vector<int64_t> &indices_shape,
                                            T *output_data, const int64_t *indices_data,
                                            const T *updates_data) const {
  int64_t indice_size = onnx_c_ops::flattened_dimension(indices_shape);
  auto n_elements = onnx_c_ops::flattened_dimension(input_shape);

  if (indice_size == 0) {
    CUDA_THROW_IF_ERROR(cudaMemsetAsync(output_data, 0, sizeof(T) * n_elements, stream));
    return;
  }

  CUDA_THROW_IF_ERROR(cudaMemsetAsync(output_data, 0, sizeof(T) * n_elements, stream));

  auto last_index_dimension = indices_shape[indices_shape.size() - 1];

  // We need element counts for each dimension and the input dim value for each dimension
  // for the range [0, last_index_dimension).
  // To avoid multiple GPU data transfers, we combine this into one array and send it through
  TensorPitches input_strides(input_shape);
  Shape2 element_counts_and_input_dims;
  memset(element_counts_and_input_dims.dims, 0, sizeof(int64_t) * last_index_dimension * 2);

  for (int64_t i = 0; i < last_index_dimension; ++i) {
    element_counts_and_input_dims.dims[i] = input_strides[i];
    element_counts_and_input_dims.dims[i + last_index_dimension] = input_shape[i];
  }

  // Let's synchronize after the initialization of the results.
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(stream));

  switch (reduction_) {
  case Reduction::Add: {
    auto element_type = CTypeToOnnxType<T>().onnx_type();
    ScatterNDImplReduction(
        stream, output_data, element_type,
        indice_size / static_cast<size_t>(last_index_dimension), indices_data,
        last_index_dimension, element_counts_and_input_dims, updates_data,
        onnx_c_ops::SizeFromDimension(input_shape, last_index_dimension, input_shape.size()),
        reduction_);
  } break;
  default:
    EXT_THROW("ScatterNDOfShape not supported for other reduction than Add, None.");
    break;
  }
}

template <typename T>
void _ComputeOptimize(cudaStream_t stream, const std::vector<int64_t> &input_shape,
                      const std::vector<int64_t> &indices_shape, T *output_data,
                      const int64_t *indices_data, const T *updates_data,
                      int maxThreadPerBlock_) {

  // The kernel is slow if there are a lot of duplicates.
  // reduction_ == Reduction::add
  // indices_shape[indices_shape.size() - 1] == 1
  // input_shape.size() == 2
  size_t indice_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(indices_shape));
  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(input_shape));
  size_t stride = input_shape[input_shape.size() - 1];
  size_t nrows = input_size / stride;

  int threads_per_block = std::min(256, maxThreadPerBlock_ / 8);

  int blocks_per_grid = (stride + threads_per_block - 1) / threads_per_block;
  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);
  addition_inplace_kernel<T><<<blocks, threads, 0, stream>>>(
      output_data, indices_data, updates_data, indice_size, nrows, stride);
}

template <typename T>
void ScatterNDOfShapeKernel<T>::ComputeOptimize(cudaStream_t &stream,
                                                const std::vector<int64_t> &input_shape,
                                                const std::vector<int64_t> &indices_shape,
                                                T *output_data, const int64_t *indices_data,
                                                const T *updates_data) const {
  _ComputeOptimize(stream, input_shape, indices_shape, output_data, indices_data, updates_data,
                   maxThreadPerBlock_);
}

static ScatterNDOfShapeOp<float> _op32;
static ScatterNDOfShapeOp<half> _op16;

} // namespace ortops
