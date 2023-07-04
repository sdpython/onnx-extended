#include "tree_ensemble.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

void* TreeEnsembleRegressor::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return std::make_unique<TreeEnsembleKernel>(api, info).release();
};

const char* TreeEnsembleRegressor::GetName() const { return "TreeEnsembleRegressor"; };

const char* TreeEnsembleRegressor::GetExecutionProviderType() const { return "CPUExecutionProvider"; };

size_t TreeEnsembleRegressor::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType TreeEnsembleRegressor::GetInputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

size_t TreeEnsembleRegressor::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType TreeEnsembleRegressor::GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };



////////////////////////
// Kernel initialization
////////////////////////

TreeEnsembleKernel::TreeEnsembleKernel(const OrtApi &api, const OrtKernelInfo *info) {
    reg_float = nullptr;
}


////////////////////////
// Kernel Implementation
////////////////////////

void TreeEnsembleKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  std::vector<int64_t> dimensions_in = input_X.GetTensorTypeAndShapeInfo().GetShape();

  Ort::UnownedValue output = ctx.GetOutput(0, dimensions_out);


  float *out = output.GetTensorMutableData<float>();
  const float *X = input_X.GetTensorData<float>();

  // Setup output, which is assumed to have the same dimensions as the inputs.


  const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
  EXT_THROW("Not implemented Yet");
}

} // namespace ortops
