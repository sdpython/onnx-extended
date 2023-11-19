#pragma once

#include "common/common_kernels.h"
#include "cpu/c_op_tfidf_vectorizer_.hpp"
// #include <memory>

namespace ortops {

template <typename TIN, typename TOUT> struct TfIdfVectorizerKernel {
  TfIdfVectorizerKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

  std::unique_ptr<onnx_c_ops::RuntimeTfIdfVectorizer<TOUT>> tfidf_typed;
};

template <typename TIN, typename TOUT>
struct TfIdfVectorizer : Ort::CustomOpBase<TfIdfVectorizer<TIN, TOUT>,
                                           TfIdfVectorizerKernel<TIN, TOUT>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

} // namespace ortops
