#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "tri_matrix.h"
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

template <typename T>
__global__ void _TriMatrixKernel(T *output_data, CUDA_LONG n_rows, CUDA_LONG n_cols, T lower,
                                 T diag, T upper) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG row = id / n_cols;
  if (row >= n_rows)
    return;
  CUDA_LONG col = id % n_cols;

  output_data[id] = (row < col) ? upper : ((row == col) ? diag : lower);
}

template <typename T>
void _TriMatrixImpl(cudaStream_t stream, int64_t n_rows, int64_t n_cols, T lower, T diag,
                    T upper, T *output_data) {
  const int n_threads = GridDim::maxThreadsPerBlock;
  CUDA_LONG N = static_cast<CUDA_LONG>(n_rows * n_cols);

  int n_blocks = (N + n_threads - 1) / n_threads;

  _TriMatrixKernel<T>
      <<<n_blocks, n_threads, 0, stream>>>(output_data, n_rows, n_cols, lower, diag, upper);
}

//////////////////
// TriMatrixOp...
//////////////////

template <typename T>
void *TriMatrixOp<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<TriMatrixKernel<T>>(api, info).release();
}

template <typename T> const char *TriMatrixOp<T>::GetName() const { return "TriMatrix"; }

template <typename T> const char *TriMatrixOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T> size_t TriMatrixOp<T>::GetInputTypeCount() const { return 2; };

template <typename T>
ONNXTensorElementDataType TriMatrixOp<T>::GetInputType(std::size_t index) const {
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case 1:
    return CTypeToOnnxType<T>().onnx_type();
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <typename T> OrtMemType TriMatrixOp<T>::GetInputMemoryType(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtMemTypeCPUInput;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <typename T>
ONNXTensorElementDataType TriMatrixOp<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
TriMatrixOp<T>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T> size_t TriMatrixOp<T>::GetOutputTypeCount() const { return 1; }

template <typename T>
OrtCustomOpInputOutputCharacteristic
TriMatrixOp<T>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// TriMatrixKernel
///////////////////

template <typename T>
TriMatrixKernel<T>::TriMatrixKernel(const OrtApi &api, const OrtKernelInfo *info) {}

template <typename T> void TriMatrixKernel<T>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 2, "Expected 2 inputs not ", n_inputs, ".");

  Ort::ConstValue shape = ctx.GetInput(0);
  Ort::ConstValue csts = ctx.GetInput(1);

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();
  // CUDA_THROW_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  auto mem = shape.GetTensorMemoryInfo();
  EXT_ENFORCE(
      mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
      "The shape should be on CPU already, but mem.GetDeviceType()=", mem.GetDeviceType(), ".");
  mem = csts.GetTensorMemoryInfo();
  EXT_ENFORCE(
      mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
      "The shape should be on CPU already, but mem.GetDeviceType()=", mem.GetDeviceType(), ".");

  std::vector<int64_t> csts_dim = csts.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(csts_dim.size() == 1 && csts_dim[0] == 3, "Second input must have three reals.")

  std::vector<int64_t> dimensions = shape.GetTensorTypeAndShapeInfo().GetShape();
  const int64_t *X = shape.GetTensorData<int64_t>();
  std::vector<int64_t> dims(X, X + dimensions[0]);
  EXT_ENFORCE(dims.size() == 2, "Shape is expected to have 2 dimensions.");
  auto output = ctx.GetOutput(0, dims);

  const T *cp = csts.GetTensorData<T>();

  _TriMatrixImpl<T>(cuda_stream, dims[0], dims[1], cp[0], cp[1], cp[2],
                    output.GetTensorMutableData<T>());
}

static TriMatrixOp<float> _kernel_32;
static TriMatrixOp<half> _kernel_16;

} // namespace ortops
