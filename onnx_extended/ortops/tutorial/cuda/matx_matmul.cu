#include "cuda/common_kernels_cuda.h"
#include "matx.h"
#include "matx_matmul.h"
#include <cublasLt.h>
#include <cublas_v2.h>

namespace ortops {

//////////////////
// MatXMatMulOp...
//////////////////

void *MatXMatMulOp::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<MatXMatMulKernel>(api, info).release();
}

const char *MatXMatMulOp::GetName() const { return op_name_; }

const char *MatXMatMulOp::GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

size_t MatXMatMulOp::GetInputTypeCount() const { return 2; };

ONNXTensorElementDataType MatXMatMulOp::GetInputType(std::size_t index) const { return dtype_; }

OrtCustomOpInputOutputCharacteristic
MatXMatMulOp::GetInputCharacteristic(std::size_t index) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

size_t MatXMatMulOp::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType MatXMatMulOp::GetOutputType(std::size_t index) const {
  return dtype_;
}

OrtCustomOpInputOutputCharacteristic
MatXMatMulOp::GetOutputCharacteristic(std::size_t index) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// MatXMatMulKernel
///////////////////

MatXMatMulKernel::MatXMatMulKernel(const OrtApi &api, const OrtKernelInfo *info) {}

void check_device(const Ort::ConstValue &input, const char *name) {
  EXT_ENFORCE(input.HasValue(), "Input '", name, "' is not empty.");
  auto mem = input.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input '", name, "' is not on CUDA");
}

void check_device(const Ort::UnownedValue &output, const char *name) {
  auto mem = output.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Output '", name, "' is not on CUDA");
}

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue &input, std::vector<int64_t> &shape,
                                          bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();
  EXT_ENFORCE(shape.size() == 2);
  if (swap) {
    std::swap(shape[0], shape[1]);
  }
  return t.GetElementType();
}

template <typename T, int RANK>
void ComputeMatMul(const std::vector<int64_t> &shape_A, const T *ptr_A,
                   const std::vector<int64_t> &shape_B, const T *ptr_B,
                   const std::vector<int64_t> &shape_D, const T *ptr_D, cudaExecutor &exec) {
  auto matx_ta = make_tensor<T, RANK>(ptr_A, shape_A);
  auto matx_tb = make_tensor<T, RANK>(ptr_B, shape_B);
  auto matx_td = make_tensor<T, RANK>(ptr_D, shape_D);
  matx_td = matmul(matx_ta, matx_tb).run(exex);
}

void MatXMatMulKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();

  check_device(input_A, "A");
  check_device(input_B, "B");
  cudaDataType_t cuda_type = ToCudaDataType(dtype_);

  std::vector<int64_t> shape_A, shape_B;
  ONNXTensorElementDataType dtype_A, dtype_B;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);
  EXT_ENFORCE(shape_A.size() == shape_B.size(), "Rank mismatch: ", shape_A.size(),
              "!=", shape_B.size(), ".");
  EXT_ENFORCE(dtype_A == dtype_, "Unexpected type for A");
  EXT_ENFORCE(dtype_B == dtype_, "Unexpected type for B");

  std::vector<int64_t> shape_D(shape_A.size());
  for (auto i = 0; i < shape_A.size() - 1; ++i)
    shape_D[i] = shape_A[i];
  shape_D[shape_D.size() - 1] = shape_B[shape_B.size() - 1];
  Ort::UnownedValue output = c.GetOutput(0, shape_D);

  cudaExecutor exec{stream};

  switch (dtype_) {
  case case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    ComputeMatMul(shape_A, input_A.GetTensorData<float>(), shape_B,
                  input_B.GetTensorData<float>(), shape_D, output.GetTensorMutableData<float>(),
                  exec);
    break;
  default:
    EXT_THROW("Not implemented for type: ", (uint64_t)dtype_, ".");
  }
}

} // namespace ortops
