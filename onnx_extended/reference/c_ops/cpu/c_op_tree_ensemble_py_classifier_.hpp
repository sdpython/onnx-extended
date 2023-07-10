#pragma once

#include "c_op_tree_ensemble_py_.hpp"
#include "cpu/c_op_tree_ensemble_common_classifier_.hpp"

namespace onnx_c_ops {

template <typename NTYPE>
class RuntimeTreeEnsembleClassifier
    : public TreeEnsembleCommonClassifier<NTYPE, NTYPE, NTYPE> {
public:
  RuntimeTreeEnsembleClassifier()
      : TreeEnsembleCommonClassifier<NTYPE, NTYPE, NTYPE>() {}
  ~RuntimeTreeEnsembleClassifier() {}

  void init(const std::string &aggregate_function, // only classifier
            py_array_t_ntype_t base_values,        // 4
            int64_t n_targets_or_classes,          // 5
            py_array_t_int64_t nodes_falsenodeids, // 6
            py_array_t_int64_t nodes_featureids,   // 7
            py_array_t_ntype_t nodes_hitrates,     // 8
            py_array_t_int64_t nodes_missing_value_tracks_true, // 9
            const std::vector<std::string> &nodes_modes,        // 10
            py_array_t_int64_t nodes_nodeids,                   // 11
            py_array_t_int64_t nodes_treeids,                   // 12
            py_array_t_int64_t nodes_truenodeids,               // 13
            py_array_t_ntype_t nodes_values,                    // 14
            const std::string &post_transform,                  // 15
            py_array_t_int64_t target_class_ids,                // 16
            py_array_t_int64_t target_class_nodeids,            // 17
            py_array_t_int64_t target_class_treeids,            // 18
            py_array_t_ntype_t target_class_weights             // 19
  ) {

    std::vector<NTYPE> cbasevalues;
    array2vector(cbasevalues, base_values, NTYPE);

    std::vector<int64_t> tnodes_treeids;
    std::vector<int64_t> tnodes_nodeids;
    std::vector<int64_t> tnodes_featureids;
    std::vector<NTYPE> tnodes_values;
    std::vector<NTYPE> tnodes_hitrates;
    std::vector<int64_t> tnodes_truenodeids;
    std::vector<int64_t> tnodes_falsenodeids;
    std::vector<int64_t> tmissing_tracks_true;

    array2vector(tnodes_falsenodeids, nodes_falsenodeids, int64_t);
    array2vector(tnodes_featureids, nodes_featureids, int64_t);
    array2vector(tnodes_hitrates, nodes_hitrates, NTYPE);
    array2vector(tmissing_tracks_true, nodes_missing_value_tracks_true,
                 int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    // nodes_modes_names_ = nodes_modes;
    array2vector(tnodes_nodeids, nodes_nodeids, int64_t);
    array2vector(tnodes_treeids, nodes_treeids, int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    array2vector(tnodes_values, nodes_values, NTYPE);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);

    std::vector<int64_t> ttarget_class_nodeids;
    std::vector<int64_t> ttarget_class_treeids;
    std::vector<int64_t> ttarget_class_ids;
    std::vector<NTYPE> ttarget_class_weights;

    array2vector(ttarget_class_ids, target_class_ids, int64_t);
    array2vector(ttarget_class_nodeids, target_class_nodeids, int64_t);
    array2vector(ttarget_class_treeids, target_class_treeids, int64_t);
    array2vector(ttarget_class_weights, target_class_weights, NTYPE);

    init_c(aggregate_function,    // 3
           cbasevalues,           // 4
           n_targets_or_classes,  // 5
           tnodes_falsenodeids,   // 6
           tnodes_featureids,     // 7
           tnodes_hitrates,       // 8
           tmissing_tracks_true,  // 9
           nodes_modes,           // 10
           tnodes_nodeids,        // 11
           tnodes_treeids,        // 12
           tnodes_truenodeids,    // 13
           tnodes_values,         // 14
           post_transform,        // 15
           ttarget_class_ids,     // 16
           ttarget_class_nodeids, // 17
           ttarget_class_treeids, // 18
           ttarget_class_weights  // 19
    );
  }

  void init_c(const std::string &aggregate_function,          // only classifier
              const std::vector<NTYPE> &base_values,          // 4
              int64_t n_targets_or_classes,                   // 5
              const std::vector<int64_t> &nodes_falsenodeids, // 6
              const std::vector<int64_t> &nodes_featureids,   // 7
              const std::vector<NTYPE> &nodes_hitrates,       // 8
              const std::vector<int64_t> &nodes_missing_value_tracks_true, // 9
              const std::vector<std::string> &nodes_modes,                 // 10
              const std::vector<int64_t> &nodes_nodeids,                   // 11
              const std::vector<int64_t> &nodes_treeids,                   // 12
              const std::vector<int64_t> &nodes_truenodeids,               // 13
              const std::vector<NTYPE> &nodes_values,                      // 14
              const std::string &post_transform,                           // 15
              const std::vector<int64_t> &target_class_ids,                // 16
              const std::vector<int64_t> &target_class_nodeids,            // 17
              const std::vector<int64_t> &target_class_treeids,            // 18
              const std::vector<NTYPE> &target_class_weights               // 19
  ) {
    this->Init(aggregate_function,              // 3
               base_values,                     // 4
               n_targets_or_classes,            // 5
               nodes_falsenodeids,              // 6
               nodes_featureids,                // 7
               nodes_hitrates,                  // 8
               nodes_missing_value_tracks_true, // 9
               nodes_modes,                     // 10
               nodes_nodeids,                   // 11
               nodes_treeids,                   // 12
               nodes_truenodeids,               // 13
               nodes_values,                    // 14
               post_transform,                  // 15
               target_class_ids,                // 16
               target_class_nodeids,            // 17
               target_class_treeids,            // 18
               target_class_weights             // 19
    );
  }

  // The two following methods uses buffers to avoid
  // spending time allocating buffers. As a consequence,
  // These methods are not thread-safe.
  py::tuple compute(py_array_t_ntype_t X) {
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
      throw std::invalid_argument("X must have 2 dimensions.");

    // Does not handle 3D tensors
    bool xdims1 = x_dims.size() == 1;
    int64_t stride = xdims1 ? x_dims[0] : x_dims[1];
    int64_t N = xdims1 ? 1 : x_dims[0];

    py_array_t_ntype_t Z(x_dims[0] * this->n_targets_or_classes_);
    py_array_t_int64_t label(x_dims[0]);

    {
      py::gil_scoped_release release;
      compute_gil_free(x_dims, N, stride, X, Z, label);
    }
    return py::make_tuple(label, Z);
  }

private:
  void compute_gil_free(const std::vector<int64_t> &x_dims, int64_t N,
                        int64_t stride, py_array_t_ntype_t &X,
                        py_array_t_ntype_t &Z, py_array_t_int64_t &label) {
    auto Z_ = _mutable_unchecked1(Z);
    auto label_ = _mutable_unchecked1(label);
    const NTYPE *x_data = X.data(0);
    NTYPE *z_data = (NTYPE *)Z_.data(0);
    int64_t *l_data = (int64_t *)label_.data(0);

    this->Compute(x_dims[0], x_dims[1], x_data, z_data, l_data);
  }
};

} // namespace onnx_c_ops
