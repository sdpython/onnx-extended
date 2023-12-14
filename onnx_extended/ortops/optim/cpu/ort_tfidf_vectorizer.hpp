#include "ort_tfidf_vectorizer.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

// Regressor

template <typename TIN, typename TOUT>
void *TfIdfVectorizer<TIN, TOUT>::CreateKernel(const OrtApi &api,
                                               const OrtKernelInfo *info) const {
  return std::make_unique<TfIdfVectorizerKernel<TIN, TOUT>>(api, info).release();
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
TfIdfVectorizerKernel<TIN, TOUT>::TfIdfVectorizerKernel(const OrtApi &api,
                                                        const OrtKernelInfo *info) {

  int64_t max_gram_length, max_skip_count, min_gram_length;
  ThrowOnError(api,
               api.KernelInfoGetAttribute_int64(info, "max_gram_length", &max_gram_length));
  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "max_skip_count", &max_skip_count));
  ThrowOnError(api,
               api.KernelInfoGetAttribute_int64(info, "min_gram_length", &min_gram_length));
  EXT_ENFORCE(max_gram_length > 0, "max_gram_length must be specifed and > 0 but is ",
              max_gram_length, ".");
  EXT_ENFORCE(max_skip_count >= 0, "max_skip_count must be specifed and >= 0 but is ",
              max_skip_count, ".");
  EXT_ENFORCE(min_gram_length > 0, "min_gram_length must be specifed and > 0 but is ",
              min_gram_length, ".");

  int64_t sparse = KernelInfoGetOptionalAttribute(api, info, "sparse", static_cast<int64_t>(0));

  std::string mode = KernelInfoGetOptionalAttributeString(api, info, "mode", "");

  std::vector<int64_t> ngram_counts =
      KernelInfoGetOptionalAttribute(api, info, "ngram_counts", std::vector<int64_t>());
  std::vector<int64_t> ngram_indexes =
      KernelInfoGetOptionalAttribute(api, info, "ngram_indexes", std::vector<int64_t>());
  std::vector<int64_t> pool_int64s =
      KernelInfoGetOptionalAttribute(api, info, "pool_int64s", std::vector<int64_t>());

  std::string pool_strings =
      KernelInfoGetOptionalAttributeString(api, info, "pool_strings", "");
  EXT_ENFORCE(pool_strings == "", "pool_strings must be empty, use pool_int64s instead.");

  std::vector<float> weights =
      KernelInfoGetOptionalAttribute(api, info, "weights", std::vector<float>());

  std::unique_ptr<onnx_c_ops::RuntimeTfIdfVectorizer<TOUT>> ptr(
      new onnx_c_ops::RuntimeTfIdfVectorizer<TOUT>());
  tfidf_typed.swap(ptr);

  tfidf_typed->Init(max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts,
                    ngram_indexes, pool_int64s, weights, sparse == 1);
}

////////////////////////
// Kernel Implementation
////////////////////////

template <typename TIN, typename TOUT>
void TfIdfVectorizerKernel<TIN, TOUT>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  std::vector<int64_t> dimensions_in = input_X.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions_in.size() <= 2, "TreeEnsemble only allows 2D inputs.");
  int64_t size = onnx_c_ops::flattened_dimension(dimensions_in);

  std::vector<TOUT> out;
  const TIN *X = input_X.GetTensorData<TIN>();
  span_type_int64 sp{X, static_cast<size_t>(size)};
  tfidf_typed->Compute(
      dimensions_in, sp, [&c = ctx](const std::vector<int64_t> &dim_out) -> span_type_tout {
        Ort::UnownedValue output = c.GetOutput(0, dim_out);
        int64_t size = onnx_c_ops::flattened_dimension(dim_out);
        return span_type_tout{output.GetTensorMutableData<TOUT>(), static_cast<size_t>(size)};
      });
}

} // namespace ortops
