#pragma once

#include "common/common_kernels.h"

namespace ortops {

struct TreeEnsembleKernel {
  TreeEnsembleRegressorKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext* context);
};

struct TreeEnsembleRegressor : Ort::CustomOpBase<TreeEnsembleRegressor, TreeEnsembleKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const ;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace ortops

