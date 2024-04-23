#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include "transpose_cast_2d.h"
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

#define TILE_DIM 32
#define BLOCK_ROWS 8

template <typename TOUT, typename TIN>
__global__ void _Transpose2DCastKernel(TOUT *output_data, const TIN *input_data, int n_rows,
                                       int n_cols) {
  __shared__ TIN tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  // int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = input_data[(y + j) * n_cols + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    output_data[(y + j) * n_rows + x] = (TOUT)(tile[threadIdx.x][threadIdx.y + j]);
}

template <typename TOUT, typename TIN>
void Transpose2DCastImpl(cudaStream_t stream, TOUT *output_data, const TIN *input_data,
                         size_t n_rows, size_t n_cols) {
  dim3 dimGrid((n_cols + TILE_DIM - 1) / TILE_DIM, (n_rows + TILE_DIM - 1) / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  _Transpose2DCastKernel<<<dimGrid, dimBlock, TILE_DIM * TILE_DIM + TILE_DIM, stream>>>(
      output_data, input_data, n_rows, n_cols);
}

//////////////////
// Transpose2DCastOp...
//////////////////

void *Transpose2DCastOp::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<Transpose2DCastKernel>(api, info).release();
}

const char *Transpose2DCastOp::GetName() const {
  switch (output_type_) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "Transpose2DCastFP16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "Transpose2DCastFP32";
  default:
    EXT_THROW("output type ", output_type_, " is not supported.");
  }
}

const char *Transpose2DCastOp::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

size_t Transpose2DCastOp::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType Transpose2DCastOp::GetInputType(std::size_t /* index */) const {
  return input_type_;
}

ONNXTensorElementDataType Transpose2DCastOp::GetOutputType(std::size_t /* index */) const {
  return output_type_;
}

OrtCustomOpInputOutputCharacteristic
Transpose2DCastOp::GetInputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

size_t Transpose2DCastOp::GetOutputTypeCount() const { return 1; }

OrtCustomOpInputOutputCharacteristic
Transpose2DCastOp::GetOutputCharacteristic(std::size_t /* index */) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// Transpose2DCastKernel
///////////////////

Transpose2DCastKernel::Transpose2DCastKernel(const OrtApi &api, const OrtKernelInfo *info) {}

void Transpose2DCastKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 1, "Expected 1 input not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::UnownedValue output;

  std::vector<int64_t> dimsA = A.GetTensorTypeAndShapeInfo().GetShape();
  auto memi = A.GetTensorMemoryInfo();
  EXT_ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "first input is not on GPU");
  EXT_ENFORCE(dimsA.size() == 2, "This operator only supports 2D tensors.")

  cudaStream_t cuda_stream = (cudaStream_t)ctx.GetGPUComputeStream();

  EXT_ENFORCE(dimsA[0] % TILE_DIM == 0 && dimsA[1] % TILE_DIM == 0,
              "This operator is implemented for dimension multiple of ", TILE_DIM,
              " but it is ", dimsA[0], "x", dimsA[1], ".");
  auto ch = dimsA[0];
  dimsA[0] = dimsA[1];
  dimsA[1] = ch;
  output = ctx.GetOutput(0, dimsA);

  auto input_type = A.GetTensorTypeAndShapeInfo().GetElementType();
  auto output_type = output.GetTensorTypeAndShapeInfo().GetElementType();
  EXT_ENFORCE(input_type != output_type, "input_type ", input_type, " and output type ",
              output_type, " should be different for Transpose2DCast.");

  size_t input_size = static_cast<size_t>(onnx_c_ops::flattened_dimension(dimsA));
  switch (input_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    switch (output_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      Transpose2DCastImpl<half, float>(cuda_stream, output.GetTensorMutableData<half>(),
                                       A.GetTensorData<float>(), dimsA[1], dimsA[0]);
      break;
    default:
      EXT_THROW("Unexpected output type ", output_type,
                " in operator Transpose2DCast (input_type=", input_type, ").");
      break;
    }
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    switch (output_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      Transpose2DCastImpl<float, half>(cuda_stream, output.GetTensorMutableData<float>(),
                                       A.GetTensorData<half>(), dimsA[1], dimsA[0]);
      break;
    default:
      EXT_THROW("Unexpected output type ", output_type,
                " in operator Transpose2DCast (input_type=", input_type, ").");
      break;
    }
    break;
  default:
    EXT_THROW("Unexpected input type ", input_type, " in operator Transpose2DCast.");
    break;
  }
}

} // namespace ortops
