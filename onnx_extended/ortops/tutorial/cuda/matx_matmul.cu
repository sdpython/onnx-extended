#include "cuda/common_kernels_cuda.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include "matx.h"
#endif
#include "matx_matmul.h"
#include <cublasLt.h>
#include <cublas_v2.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
using namespace matx;
#endif

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

static void check_device(const Ort::ConstValue &input, const char *name) {
  EXT_ENFORCE(input.HasValue(), "Input '", name, "' is not empty.");
  auto mem = input.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input '", name, "' is not on CUDA");
}

static void check_device(const Ort::UnownedValue &output, const char *name) {
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

template <typename T>
void ComputeMatMul(const std::vector<int64_t> &shape_A, const T *ptr_A,
                   const std::vector<int64_t> &shape_B, const T *ptr_B,
                   const std::vector<int64_t> &shape_D, const T *ptr_D, cudaStream_t &stream) {
  // MatX only supports tensors with a known ranks.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  if (shape_A.size() == 2 && shape_B.size() == 2) {
    auto matx_ta = make_tensor(ptr_A, {shape_A[0], shape_A[1]});
    auto matx_tb = make_tensor(ptr_B, {shape_B[0], shape_B[1]});
    auto matx_td = make_tensor<T>({shape_D[0], shape_D[1]});
    (matx_td = matmul(matx_ta, matx_tb)).run(stream);
    CUDA_THROW_IF_ERROR(cudaMemcpyAsync((void *)ptr_D, (void *)matx_td.data(),
                                        sizeof(T) * shape_D[0] * shape_D[1],
                                        cudaMemcpyDeviceToDevice));
  } else {
    EXT_THROW("ComputeMatMul not implemented when ranks are ", shape_A.size(), " and ",
              shape_B.size(), ".");
  }
#else
  EXT_THROW("ComputeMatMul not implemented with CUDA_VERSION=", CUDA_VERSION, ".");
#endif
}

void MatXMatMulKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();

  check_device(input_A, "A");
  check_device(input_B, "B");

  std::vector<int64_t> shape_A, shape_B;
  ONNXTensorElementDataType dtype_A, dtype_B;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);
  EXT_ENFORCE(shape_A.size() == shape_B.size(), "Rank mismatch: ", shape_A.size(),
              "!=", shape_B.size(), ".");
  EXT_ENFORCE(dtype_A == dtype_B, "Unexpected type for A or B");
  cudaDataType_t cuda_type = ToCudaDataType(dtype_A);

  std::vector<int64_t> shape_D(shape_A.size());
  for (auto i = 0; i < shape_A.size() - 1; ++i)
    shape_D[i] = shape_A[i];
  shape_D[shape_D.size() - 1] = shape_B[shape_B.size() - 1];
  Ort::UnownedValue output = ctx.GetOutput(0, shape_D);
  check_device(output, "Y");

  switch (dtype_A) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    ComputeMatMul(shape_A, input_A.GetTensorData<float>(), shape_B,
                  input_B.GetTensorData<float>(), shape_D, output.GetTensorMutableData<float>(),
                  stream);
    break;
  default:
    EXT_THROW("Not implemented for type: ", (uint64_t)dtype_A, ".");
  }
}

} // namespace ortops
