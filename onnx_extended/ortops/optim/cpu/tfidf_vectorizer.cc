#include "tfidf_vectorizer.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

// Regressor

template <typename TIN, typename TOUTIN, typename TIN, typename TOUTOUT>
void *
TfIdfVectorizer<TIN, TOUT>::CreateKernel(const OrtApi &api,
                                         const OrtKernelInfo *info) const {
  return std::make_unique<TfIdfVectorizerKernel<TIN, TOUT>>(api, info)
      .release();
};

template <> const char *TfIdfVectorizer<int64_t, float>::GetName() const {
  return "TfIdfVectorizer";
};

template <typename TIN, typename TOUT>
const char *TfIdfVectorizer<TIN, TOUT>::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

template <typename TIN, typename TOUT>
size_t TfIdfVectorizer<TIN, TOUT>::GetInputTypeCount() const {
  return 1;
};

template <>
ONNXTensorElementDataType
TfIdfVectorizer<int64_t, float>::GetInputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

template <typename TIN, typename TOUT>
size_t TfIdfVectorizer<TIN, TOUT>::GetOutputTypeCount() const {
  return 1;
};

template <>
ONNXTensorElementDataType
TfIdfVectorizer<int64_t, float>::GetOutputType(std::size_t /* index */) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

////////////////////////
// Kernel initialization
////////////////////////

template <typename TIN, typename TOUT>
TfIdfVectorizerKernel<TIN, TOUT>::TfIdfVectorizerKernel(
    const OrtApi &api, const OrtKernelInfo *info) {}

////////////////////////
// Kernel Implementation
////////////////////////

template <typename TIN, typename TOUT>
void TfIdfVectorizerKernel<TIN, TOUT>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  std::vector<int64_t> dimensions_in =
      input_X.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions_in.size() <= 2, "TreeEnsemble only allows 2D inputs.");
  std::vector<int64_t> dimensions_out{dimensions_in[0], n_targets_or_classes};
  // Ort::UnownedValue output = ctx.GetOutput(is_classifier ? 1 : 0,
  // dimensions_out);
}

} // namespace ortops
