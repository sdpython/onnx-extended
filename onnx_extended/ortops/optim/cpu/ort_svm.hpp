#pragma once

#include "ort_svm.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

// Regressor

template <typename T>
void *SVMRegressor<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<SVMKernel<T>>(api, info).release();
};

template <> const char *SVMRegressor<float>::GetName() const { return "SVMRegressor"; };

template <typename T> const char *SVMRegressor<T>::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

template <typename T> size_t SVMRegressor<T>::GetInputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType SVMRegressor<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
};

template <typename T> size_t SVMRegressor<T>::GetOutputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType SVMRegressor<T>::GetOutputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
};

// Classifier

template <typename T>
void *SVMClassifier<T>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<SVMKernel<T>>(api, info).release();
};

template <> const char *SVMClassifier<float>::GetName() const { return "SVMClassifier"; };

template <typename T> const char *SVMClassifier<T>::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

template <typename T> size_t SVMClassifier<T>::GetInputTypeCount() const { return 1; };

template <typename T>
ONNXTensorElementDataType SVMClassifier<T>::GetInputType(std::size_t /* index */) const {
  return CTypeToOnnxType<T>().onnx_type();
};

template <typename T> size_t SVMClassifier<T>::GetOutputTypeCount() const { return 2; };

template <typename T>
ONNXTensorElementDataType SVMClassifier<T>::GetOutputType(std::size_t index) const {
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case 1:
    return CTypeToOnnxType<T>().onnx_type();
  default:
    EXT_THROW("Unexpected output index: ", (uint64_t)index, ".");
  }
};

////////////////////////
// Kernel initialization
////////////////////////

template <typename T> SVMKernel<T>::SVMKernel(const OrtApi &api, const OrtKernelInfo *info) {
  svm_type = nullptr;

  std::vector<float> coefficients =
      KernelInfoGetOptionalAttribute(api, info, "coefficients", std::vector<float>());
  std::vector<float> kernel_params =
      KernelInfoGetOptionalAttribute(api, info, "kernel_params", std::vector<float>());
  std::string kernel_type = KernelInfoGetOptionalAttributeString(api, info, "kernel_type", "");
  std::string post_transform =
      KernelInfoGetOptionalAttributeString(api, info, "post_transform", "");
  std::vector<float> rho =
      KernelInfoGetOptionalAttribute(api, info, "rho", std::vector<float>());
  std::vector<float> support_vectors =
      KernelInfoGetOptionalAttribute(api, info, "support_vectors", std::vector<float>());

  int64_t n_supports =
      KernelInfoGetOptionalAttribute(api, info, "n_supports", static_cast<int64_t>(-1));

  int64_t one_class = -1;
  std::vector<float> prob_a, prob_b;
  std::vector<int64_t> classlabels_ints, vectors_per_class;

  if (n_supports == -1) {
    // A classifier.
    is_classifier = true;

    prob_a = KernelInfoGetOptionalAttribute(api, info, "prob_a", std::vector<float>());
    prob_b = KernelInfoGetOptionalAttribute(api, info, "prob_b", std::vector<float>());
    classlabels_ints =
        KernelInfoGetOptionalAttribute(api, info, "classlabels_ints", std::vector<int64_t>());
    vectors_per_class =
        KernelInfoGetOptionalAttribute(api, info, "vectors_per_class", std::vector<int64_t>());
    n_targets_or_classes = classlabels_ints.size();
  } else {
    // A regressor.
    is_classifier = false;
    one_class = KernelInfoGetOptionalAttribute(api, info, "one_class", static_cast<int64_t>(0));
    EXT_ENFORCE(one_class >= 0, "unexpected value for one_class=", one_class, ".");
  }

  std::unique_ptr<onnx_c_ops::RuntimeSVMCommon<T>> ptr(new onnx_c_ops::RuntimeSVMCommon<T>());
  svm_type.swap(ptr);
  svm_type->init(coefficients, kernel_params, kernel_type, post_transform, rho, support_vectors,
                 // regressor
                 n_supports, one_class,
                 // classifier
                 prob_a, prob_b, classlabels_ints, vectors_per_class);
  if (!is_classifier) {
    n_targets_or_classes = 1; // what about multiregression?
  }
}

////////////////////////
// Kernel Implementation
////////////////////////

template <typename T> void SVMKernel<T>::Compute(OrtKernelContext *context) {

  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  std::vector<int64_t> dimensions_in = input_X.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions_in.size() == 2, "TreeEnsemble only allows 2D inputs.");

  int64_t *p_labels = nullptr;
  const T *X = input_X.GetTensorData<T>();
  int64_t stride = dimensions_in.size() == 1 ? dimensions_in[0] : dimensions_in[1];

  if (is_classifier) {
    std::vector<int64_t> dimensions_label{dimensions_in[0]};
    Ort::UnownedValue labels = ctx.GetOutput(0, dimensions_label);
    p_labels = labels.GetTensorMutableData<int64_t>();
    std::vector<int64_t> dimensions_out{dimensions_in[0], n_targets_or_classes};
    Ort::UnownedValue output = ctx.GetOutput(1, dimensions_out);

    EXT_ENFORCE(svm_type.get() != nullptr, "No implementation yet for input type=",
                (uint64_t)input_X.GetTensorTypeAndShapeInfo().GetElementType(),
                " and output type=",
                (uint64_t)output.GetTensorTypeAndShapeInfo().GetElementType(), ".");

    T *out = output.GetTensorMutableData<T>();
    svm_type->compute_classifier(dimensions_in, dimensions_in[0], stride, X, p_labels, out,
                                 svm_type->get_n_columns());
  } else {
    std::vector<int64_t> dimensions_out{dimensions_in[0], n_targets_or_classes};
    Ort::UnownedValue output = ctx.GetOutput(0, dimensions_out);
    EXT_ENFORCE(svm_type.get() != nullptr, "No implementation yet for input type=",
                (uint64_t)input_X.GetTensorTypeAndShapeInfo().GetElementType(),
                " and output type=",
                (uint64_t)output.GetTensorTypeAndShapeInfo().GetElementType(), ".");

    T *out = output.GetTensorMutableData<T>();
    svm_type->compute_regressor(dimensions_in, dimensions_in[0], stride, X, out);
  }
}

} // namespace ortops
