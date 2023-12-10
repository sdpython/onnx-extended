#include "my_kernel.h"

namespace ortops {

MyCustomKernel::MyCustomKernel(const OrtApi & /* api */, const OrtKernelInfo * /* info */) {}

void MyCustomKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  Ort::ConstValue input_Y = ctx.GetInput(1);
  const float *X = input_X.GetTensorData<float>();
  const float *Y = input_Y.GetTensorData<float>();

  // Setup output, which is assumed to have the same dimensions as the inputs.
  std::vector<int64_t> dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

  Ort::UnownedValue output = ctx.GetOutput(0, dimensions);
  float *out = output.GetTensorMutableData<float>();

  const std::size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

  // Do computation
  for (std::size_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
}

void *MyCustomOp::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<MyCustomKernel>(api, info).release();
}

const char *MyCustomOp::GetName() const { return "MyCustomOp"; };

const char *MyCustomOp::GetExecutionProviderType() const { return "CPUExecutionProvider"; }

size_t MyCustomOp::GetInputTypeCount() const { return 2; };

ONNXTensorElementDataType MyCustomOp::GetInputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t MyCustomOp::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType MyCustomOp::GetOutputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

} // namespace ortops
