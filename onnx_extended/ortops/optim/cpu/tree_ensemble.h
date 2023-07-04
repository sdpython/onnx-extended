#pragma once

#include "common/common_kernels.h"
#include "cpu/c_op_tree_ensemble_common_.hpp"
// #include <memory>

namespace ortops {

struct TreeEnsembleKernel {
  TreeEnsembleKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

  // Attributes
  std::unique_ptr<onnx_c_ops::TreeEnsembleCommon<float, float, float>>
      reg_float_float_float;
};

struct TreeEnsembleRegressor
    : Ort::CustomOpBase<TreeEnsembleRegressor, TreeEnsembleKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace ortops
