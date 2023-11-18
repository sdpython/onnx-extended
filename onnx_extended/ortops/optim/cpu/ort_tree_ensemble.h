#pragma once

#include "common/common_kernels.h"
#include "cpu/c_op_tree_ensemble_common_.hpp"
// #include <memory>

namespace ortops {

template <typename ITYPE, typename TTYPE, typename OTYPE>
struct TreeEnsembleKernel {
  TreeEnsembleKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

  // Attributes
  int64_t n_targets_or_classes;
  std::unique_ptr<onnx_c_ops::TreeEnsembleCommon<ITYPE, TTYPE, OTYPE>>
      reg_type_type_type;
  bool is_classifier;
};

template <typename ITYPE, typename TTYPE, typename OTYPE>
struct TreeEnsembleRegressor
    : Ort::CustomOpBase<TreeEnsembleRegressor<ITYPE, TTYPE, OTYPE>,
                        TreeEnsembleKernel<ITYPE, TTYPE, OTYPE>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

template <typename ITYPE, typename TTYPE, typename OTYPE>
struct TreeEnsembleClassifier
    : Ort::CustomOpBase<TreeEnsembleClassifier<ITYPE, TTYPE, OTYPE>,
                        TreeEnsembleKernel<ITYPE, TTYPE, OTYPE>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

} // namespace ortops
