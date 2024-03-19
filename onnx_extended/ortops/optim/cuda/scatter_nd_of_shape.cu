#include "cuda/common_kernels_cuda.h"
#include "scatter_nd_of_shape.h"
#include <chrono>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

namespace ortops {

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
  case 2:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<half>::GetInputType(std::size_t index) const {
  switch (index) {
  case 0:
  case 2:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
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
  std::string reduction = value_string;
  if (reduction == "add")
    reduction_ = Reduction::Add;
  else
    EXT_THROW("unexpected reduction '", reduction, "'.");
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
  std::vector<int64_t> update_shape = updates.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions.size() == 1, "shape must be a 1-dimension tensor.");

  auto mem = shape.GetTensorMemoryInfo();
  if (mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU) {
    cudaDeviceSynchronize();
    std::vector<int64_t> buf(dimensions[0]);
    const int64_t *ptr = shape.GetTensorData<int64_t>();
    cudaMemcpy(buf.data(), ptr, dimensions[0] * sizeof(int64_t), cudaMemcpyDeviceToHost);
    output = ctx.GetOutput(0, buf);
  } else if (mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU) {
    const int64_t *X = shape.GetTensorData<int64_t>();
    std::vector<int64_t> dims(dimensions[0]);
    for (size_t i = 0; i < dimensions[0]; ++i)
      dims[i] = X[i];
    output = ctx.GetOutput(0, dimensions);
  } else {
    EXT_THROW("Unexpected device for input 1.");
  }
}

} // namespace ortops
