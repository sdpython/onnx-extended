#pragma once

// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "c_op_tree_ensemble_common_p_agg_.hpp"
#include <omp.h>

// https://cims.nyu.edu/~stadler/hpc17/material/ompLec.pdf
// http://amestoy.perso.enseeiht.fr/COURS/CoursMulticoreProgrammingButtari.pdf

#if 0
#define DEBUGPRINT(s) std::cout << MakeString(s, "\n");
#define ASSERTTREE(cond, msg)                                                  \
  if (!(cond))                                                                 \
    throw std::invalid_argument(MakeString(msg, " - failed: ", #cond));
#else
#define DEBUGPRINT(s)
#define ASSERTTREE(cond, msg)
#endif

namespace onnx_c_ops {

#define py_array_t_int64_t                                                     \
  py::array_t<int64_t, py::array::c_style | py::array::forcecast>
#define py_array_t_ntype_t                                                     \
  py::array_t<NTYPE, py::array::c_style | py::array::forcecast>

/**
 * This classes parallelizes itself the computation,
 * it keeps buffer for every thread it generates. Calling
 * the same compute function from different thread will
 * cause computation errors. The class is not thread safe.
 */
template <typename NTYPE>
class RuntimeTreeEnsembleCommonP
    : public TreeEnsembleCommon<NTYPE, NTYPE, NTYPE> {
public:
  RuntimeTreeEnsembleCommonP(int omp_tree, int omp_tree_N, int omp_N);
  ~RuntimeTreeEnsembleCommonP();

  void init(const std::string &aggregate_function, // only classifier
            py_array_t_ntype_t base_values, int64_t n_targets_or_classes,
            py_array_t_int64_t nodes_falsenodeids,
            py_array_t_int64_t nodes_featureids,
            py_array_t_ntype_t nodes_hitrates,
            py_array_t_int64_t nodes_missing_value_tracks_true,
            const std::vector<std::string> &nodes_modes,
            py_array_t_int64_t nodes_nodeids, py_array_t_int64_t nodes_treeids,
            py_array_t_int64_t nodes_truenodeids,
            py_array_t_ntype_t nodes_values, const std::string &post_transform,
            py_array_t_int64_t target_class_ids,
            py_array_t_int64_t target_class_nodeids,
            py_array_t_int64_t target_class_treeids,
            py_array_t_ntype_t target_class_weights);

  void init_c(const std::string &aggregate_function, // only classifier
              const std::vector<NTYPE> &base_values,
              int64_t n_targets_or_classes,
              const std::vector<int64_t> &nodes_falsenodeids,
              const std::vector<int64_t> &nodes_featureids,
              const std::vector<NTYPE> &nodes_hitrates,
              const std::vector<int64_t> &nodes_missing_value_tracks_true,
              const std::vector<std::string> &nodes_modes,
              const std::vector<int64_t> &nodes_nodeids,
              const std::vector<int64_t> &nodes_treeids,
              const std::vector<int64_t> &nodes_truenodeids,
              const std::vector<NTYPE> &nodes_values,
              const std::string &post_transform,
              const std::vector<int64_t> &target_class_ids,
              const std::vector<int64_t> &target_class_nodeids,
              const std::vector<int64_t> &target_class_treeids,
              const std::vector<NTYPE> &target_class_weights);

  template <typename AGG>
  py::array_t<NTYPE> compute_tree_outputs_agg(py_array_t_ntype_t X,
                                              const AGG &agg) const;

  // The two following methods uses buffers to avoid
  // spending time allocating buffers. As a consequence,
  // These methods are not thread-safe.
  template <typename AGG>
  py::array_t<NTYPE> compute_agg(py_array_t_ntype_t X, const AGG &agg);

  template <typename AGG>
  py::tuple compute_cl_agg(py_array_t_ntype_t X, const AGG &agg);

private:
  template <typename AGG>
  void compute_gil_free(const std::vector<int64_t> &x_dims, int64_t N,
                        int64_t stride, py_array_t_ntype_t &X,
                        py_array_t_ntype_t &Z, py_array_t_int64_t *Y,
                        const AGG &agg);

  template <typename AGG>
  void compute_gil_free_array_structure(const std::vector<int64_t> &x_dims,
                                        int64_t N, int64_t stride,
                                        py_array_t_ntype_t &X,
                                        py_array_t_ntype_t &Z,
                                        py_array_t_int64_t *Y, const AGG &agg);

  void switch_to_array_structure();
};

template <typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::RuntimeTreeEnsembleCommonP(int omp_tree,
                                                              int omp_tree_N,
                                                              int omp_N) {
  omp_tree_ = omp_tree;
  omp_tree_N_ = omp_tree_N;
  omp_N_ = omp_N;
}

template <typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::~RuntimeTreeEnsembleCommonP() {}

template <typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init(
    const std::string &aggregate_function, py_array_t_ntype_t base_values,
    int64_t n_targets_or_classes, py_array_t_int64_t nodes_falsenodeids,
    py_array_t_int64_t nodes_featureids, py_array_t_ntype_t nodes_hitrates,
    py_array_t_int64_t nodes_missing_value_tracks_true,
    const std::vector<std::string> &nodes_modes,
    py_array_t_int64_t nodes_nodeids, py_array_t_int64_t nodes_treeids,
    py_array_t_int64_t nodes_truenodeids, py_array_t_ntype_t nodes_values,
    const std::string &post_transform, py_array_t_int64_t target_class_ids,
    py_array_t_int64_t target_class_nodeids,
    py_array_t_int64_t target_class_treeids,
    py_array_t_ntype_t target_class_weights) {

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
  array2vector(tmissing_tracks_true, nodes_missing_value_tracks_true, int64_t);
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

  init_c(aggregate_function, cbasevalues, n_targets_or_classes,
         tnodes_falsenodeids, tnodes_featureids, tnodes_hitrates,
         tmissing_tracks_true, nodes_modes, tnodes_nodeids, tnodes_treeids,
         tnodes_truenodeids, tnodes_values, post_transform, ttarget_class_ids,
         ttarget_class_nodeids, ttarget_class_treeids, ttarget_class_weights);
}

py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(
    py::array_t<float, py::array::c_style | py::array::forcecast> &Z) {
  return Z.mutable_unchecked<1>();
}

py::detail::unchecked_mutable_reference<int64_t, 1>
_mutable_unchecked1(py_array_t_int64_t &Z) {
  return Z.mutable_unchecked<1>();
}

py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(
    py::array_t<double, py::array::c_style | py::array::forcecast> &Z) {
  return Z.mutable_unchecked<1>();
}

template <typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init_c(
    const std::string &aggregate_function,
    const std::vector<NTYPE> &base_values, int64_t n_targets_or_classes,
    const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_featureids,
    const std::vector<NTYPE> &nodes_hitrates,
    const std::vector<int64_t> &nodes_missing_value_tracks_true,
    const std::vector<std::string> &nodes_modes,
    const std::vector<int64_t> &nodes_nodeids,
    const std::vector<int64_t> &nodes_treeids,
    const std::vector<int64_t> &nodes_truenodeids,
    const std::vector<NTYPE> &nodes_values, const std::string &post_transform,
    const std::vector<int64_t> &target_class_ids,
    const std::vector<int64_t> &target_class_nodeids,
    const std::vector<int64_t> &target_class_treeids,
    const std::vector<NTYPE> &target_class_weights) {}

template <typename NTYPE>
template <typename AGG>
py::array_t<NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::compute_agg(py_array_t_ntype_t X,
                                               const AGG &agg) {
  std::vector<int64_t> x_dims;
  arrayshape2vector(x_dims, X);
  if (x_dims.size() != 2)
    throw std::invalid_argument("X must have 2 dimensions.");

  // Does not handle 3D tensors
  bool xdims1 = x_dims.size() == 1;
  int64_t stride = xdims1 ? x_dims[0] : x_dims[1];
  int64_t N = xdims1 ? 1 : x_dims[0];

  py_array_t_ntype_t Z(x_dims[0] * n_targets_or_classes_);

  {
    py::gil_scoped_release release;
    compute_gil_free(x_dims, N, stride, X, Z, nullptr, agg);
  }
  return Z;
}

template <typename NTYPE>
template <typename AGG>
py::tuple
RuntimeTreeEnsembleCommonP<NTYPE>::compute_cl_agg(py_array_t_ntype_t X,
                                                  const AGG &agg) {
  std::vector<int64_t> x_dims;
  arrayshape2vector(x_dims, X);
  if (x_dims.size() != 2)
    throw std::invalid_argument("X must have 2 dimensions.");

  // Does not handle 3D tensors
  bool xdims1 = x_dims.size() == 1;
  int64_t stride = xdims1 ? x_dims[0] : x_dims[1];
  int64_t N = xdims1 ? 1 : x_dims[0];

  // Tensor* Y = context->Output(0, TensorShape({N}));
  // auto* Z = context->Output(1, TensorShape({N, class_count_}));
  py_array_t_ntype_t Z(x_dims[0] * n_targets_or_classes_);
  py_array_t_int64_t Y(x_dims[0]);

  {
    py::gil_scoped_release release;
    compute_gil_free(x_dims, N, stride, X, Z, &Y, agg);
  }
  return py::make_tuple(Y, Z);
}

template <typename NTYPE>
template <typename AGG>
void RuntimeTreeEnsembleCommonP<NTYPE>::compute_gil_free(
    const std::vector<int64_t> &x_dims, int64_t N, int64_t stride,
    const py::array_t<NTYPE, py::array::c_style | py::array::forcecast> &X,
    py_array_t_ntype_t &Z, py_array_t_int64_t *Y, const AGG &agg) {

  // expected primary-expression before ')' token
  auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
  const NTYPE *x_data = X.data(0);
  auto nth = omp_get_max_threads();
}

} // namespace onnx_c_ops
