#include "tree_ensemble.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

// Regressor

void *TreeEnsembleRegressor::CreateKernel(const OrtApi &api,
                                          const OrtKernelInfo *info) const {
  return std::make_unique<TreeEnsembleKernel>(api, info).release();
};

const char *TreeEnsembleRegressor::GetName() const {
  return "TreeEnsembleRegressor";
};

const char *TreeEnsembleRegressor::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

size_t TreeEnsembleRegressor::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType
TreeEnsembleRegressor::GetInputType(std::size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t TreeEnsembleRegressor::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType
TreeEnsembleRegressor::GetOutputType(std::size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

// Classifier

void *TreeEnsembleClassifier::CreateKernel(const OrtApi &api,
                                           const OrtKernelInfo *info) const {
  return std::make_unique<TreeEnsembleKernel>(api, info).release();
};

const char *TreeEnsembleClassifier::GetName() const {
  return "TreeEnsembleClassifier";
};

const char *TreeEnsembleClassifier::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
};

size_t TreeEnsembleClassifier::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType
TreeEnsembleClassifier::GetInputType(std::size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t TreeEnsembleClassifier::GetOutputTypeCount() const { return 2; };

ONNXTensorElementDataType
TreeEnsembleClassifier::GetOutputType(std::size_t index) const {
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("Unexpected output index: ", (uint64_t)index, ".");
  }
};

////////////////////////
// Kernel initialization
////////////////////////

TreeEnsembleKernel::TreeEnsembleKernel(const OrtApi &api,
                                       const OrtKernelInfo *info) {
  reg_float_float_float = nullptr;

  std::string aggregate_function = KernelInfoGetOptionalAttributeString(
      api, info, "aggregate_function", "SUM");
  std::vector<float> base_values = KernelInfoGetOptionalAttribute(
      api, info, "base_values", std::vector<float>());
  n_targets_or_classes = KernelInfoGetOptionalAttribute(
      api, info, "n_targets", static_cast<int64_t>(1));
  std::vector<int64_t> nodes_falsenodeids = KernelInfoGetOptionalAttribute(
      api, info, "nodes_falsenodeids", std::vector<int64_t>());
  std::vector<int64_t> nodes_featureids = KernelInfoGetOptionalAttribute(
      api, info, "nodes_featureids", std::vector<int64_t>());
  std::vector<float> nodes_hitrates = KernelInfoGetOptionalAttribute(
      api, info, "nodes_hitrates", std::vector<float>());
  std::vector<int64_t> nodes_missing_value_tracks_true =
      KernelInfoGetOptionalAttribute(
          api, info, "nodes_missing_value_tracks_true", std::vector<int64_t>());
  std::string nodes_modes_single =
      KernelInfoGetOptionalAttributeString(api, info, "nodes_modes", "");
  std::vector<int64_t> nodes_nodeids = KernelInfoGetOptionalAttribute(
      api, info, "nodes_nodeids", std::vector<int64_t>());
  std::vector<int64_t> nodes_treeids = KernelInfoGetOptionalAttribute(
      api, info, "nodes_treeids", std::vector<int64_t>());
  std::vector<int64_t> nodes_truenodeids = KernelInfoGetOptionalAttribute(
      api, info, "nodes_truenodeids", std::vector<int64_t>());
  std::vector<float> nodes_values = KernelInfoGetOptionalAttribute(
      api, info, "nodes_values", std::vector<float>());
  std::string post_transform =
      KernelInfoGetOptionalAttributeString(api, info, "post_transform", "NONE");

  std::vector<int64_t> target_class_nodeids = KernelInfoGetOptionalAttribute(
      api, info, "target_nodeids", std::vector<int64_t>());
  std::vector<int64_t> target_class_ids;
  std::vector<int64_t> target_class_treeids;
  std::vector<float> target_class_weights;
  if (target_class_nodeids.empty()) {
    // A classifier.
    target_class_nodeids = KernelInfoGetOptionalAttribute(
        api, info, "class_nodeids", std::vector<int64_t>());
    target_class_ids = KernelInfoGetOptionalAttribute(api, info, "class_ids",
                                                      std::vector<int64_t>());
    target_class_treeids = KernelInfoGetOptionalAttribute(
        api, info, "class_treeids", std::vector<int64_t>());
    target_class_weights = KernelInfoGetOptionalAttribute(
        api, info, "class_weights", std::vector<float>());
    is_classifier = true;
    std::vector<int64_t> labels_ints = KernelInfoGetOptionalAttribute(
        api, info, "classlabels_int64s", std::vector<int64_t>());
    EXT_ENFORCE(!labels_ints.empty(),
                "This kernel does not support string classes.");
    n_targets_or_classes = labels_ints.size();
    for (std::size_t i = 0; i < labels_ints.size(); ++i) {
      EXT_ENFORCE(labels_ints[i] == i,
                  "classlabels_int64s should be an array of consecutive "
                  "integers starting at 0, but position ",
                  (uint64_t)i, " fails.");
    }
  } else {
    // A regressor.
    target_class_ids = KernelInfoGetOptionalAttribute(api, info, "target_ids",
                                                      std::vector<int64_t>());
    target_class_treeids = KernelInfoGetOptionalAttribute(
        api, info, "target_treeids", std::vector<int64_t>());
    target_class_weights = KernelInfoGetOptionalAttribute(
        api, info, "target_weights", std::vector<float>());
    is_classifier = false;
  }

  std::vector<std::string> nodes_modes = SplitString(nodes_modes_single, ',');
  EXT_ENFORCE(n_targets_or_classes > 0);
  EXT_ENFORCE(nodes_values.size() > 0);
  EXT_ENFORCE(nodes_nodeids.size() > 0);
  EXT_ENFORCE(nodes_modes.size() == nodes_falsenodeids.size(),
              " nodes_modes.size()==", (uint64_t)nodes_modes.size(),
              "!=", (uint64_t)nodes_falsenodeids.size(),
              ", nodes_modes=", nodes_modes_single, ".");
  EXT_ENFORCE(n_targets_or_classes > 0);

  std::unique_ptr<onnx_c_ops::TreeEnsembleCommon<float, float, float>> ptr(
      new onnx_c_ops::TreeEnsembleCommon<float, float, float>());
  reg_float_float_float.swap(ptr);
  auto status = reg_float_float_float->Init(
      aggregate_function, base_values, n_targets_or_classes, nodes_falsenodeids,
      nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true,
      nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids,
      nodes_values, post_transform, target_class_ids, target_class_nodeids,
      target_class_treeids, target_class_weights);
  EXT_ENFORCE(status.IsOK(), "The tree ensemble initialisation failed.");

  int64_t parallel_tree = KernelInfoGetOptionalAttribute(
      api, info, "parallel_tree", static_cast<int64_t>(80));
  int64_t parallel_tree_N = KernelInfoGetOptionalAttribute(
      api, info, "parallel_tree_N", static_cast<int64_t>(128));
  int64_t parallel_N = KernelInfoGetOptionalAttribute(api, info, "parallel_N",
                                                      static_cast<int64_t>(50));
  int64_t batch_size_tree = KernelInfoGetOptionalAttribute(
      api, info, "batch_size_tree", static_cast<int64_t>(2));
  int64_t batch_size_rows = KernelInfoGetOptionalAttribute(
      api, info, "batch_size_rows", static_cast<int64_t>(2));
  int64_t use_node3 = KernelInfoGetOptionalAttribute(api, info, "use_node3",
                                                     static_cast<int64_t>(0));

  reg_float_float_float->set(parallel_tree, parallel_tree_N, parallel_N,
                             batch_size_tree, batch_size_rows, use_node3);
}

////////////////////////
// Kernel Implementation
////////////////////////

void TreeEnsembleKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  std::vector<int64_t> dimensions_in =
      input_X.GetTensorTypeAndShapeInfo().GetShape();
  EXT_ENFORCE(dimensions_in.size() == 2, "TreeEnsemble only allows 2D inputs.");
  std::vector<int64_t> dimensions_out{dimensions_in[0], n_targets_or_classes};
  Ort::UnownedValue output = ctx.GetOutput(is_classifier ? 1 : 0, dimensions_out);
  int64_t *p_labels = nullptr;
  if (is_classifier) {
    std::vector<int64_t> dimensions_label{dimensions_in[0]};
    Ort::UnownedValue labels = ctx.GetOutput(0, dimensions_label);
    p_labels = labels.GetTensorMutableData<int64_t>();
  }

  if (reg_float_float_float.get() != nullptr) {
    const float *X = input_X.GetTensorData<float>();
    float *out = output.GetTensorMutableData<float>();
    reg_float_float_float->Compute(dimensions_in[0], dimensions_in[1], X, out,
                                   p_labels);
  } else {
    EXT_ENFORCE("No implementation yet for input type=",
                (uint64_t)input_X.GetTensorTypeAndShapeInfo().GetElementType(),
                " and output type=",
                (uint64_t)output.GetTensorTypeAndShapeInfo().GetElementType(),
                ".");
  }
}

} // namespace ortops
