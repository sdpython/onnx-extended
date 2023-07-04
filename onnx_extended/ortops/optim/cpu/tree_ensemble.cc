#include "tree_ensemble.h"

namespace ortops {

////////////////////////
// Operators declaration
////////////////////////

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
TreeEnsembleRegressor::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t TreeEnsembleRegressor::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType
TreeEnsembleRegressor::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
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
  int64_t n_targets_or_classes = KernelInfoGetOptionalAttribute(
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
  std::vector<int64_t> target_class_ids = KernelInfoGetOptionalAttribute(
      api, info, "target_class_ids", std::vector<int64_t>());
  std::vector<int64_t> target_class_nodeids = KernelInfoGetOptionalAttribute(
      api, info, "target_class_nodeids", std::vector<int64_t>());
  std::vector<int64_t> target_class_treeids = KernelInfoGetOptionalAttribute(
      api, info, "target_class_treeids", std::vector<int64_t>());
  std::vector<float> target_class_weights = KernelInfoGetOptionalAttribute(
      api, info, "target_class_weights", std::vector<float>());

  std::vector<std::string> nodes_modes = SplitString(nodes_modes_single, ',');

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

  /*
  Ort::UnownedValue output = ctx.GetOutput(0, dimensions_out);


  float *out = output.GetTensorMutableData<float>();
  const float *X = input_X.GetTensorData<float>();

  // Setup output, which is assumed to have the same dimensions as the inputs.


  const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
  */
  EXT_THROW("Not implemented Yet");
}

} // namespace ortops
