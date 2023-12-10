#pragma once

#include "common/common_kernels.h"
#include "cpu/c_op_svm_common_.hpp"

namespace ortops {

template <typename T> struct SVMKernel {
  SVMKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

  // Attributes
  int64_t n_targets_or_classes;
  std::unique_ptr<onnx_c_ops::RuntimeSVMCommon<T>> svm_type;
  bool is_classifier;
};

template <typename T> struct SVMRegressor : Ort::CustomOpBase<SVMRegressor<T>, SVMKernel<T>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

template <typename T> struct SVMClassifier : Ort::CustomOpBase<SVMClassifier<T>, SVMKernel<T>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

} // namespace ortops
