// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Classifier.cc.

#include "c_op_tree_ensemble_py_.hpp"
#include "c_op_tree_ensemble_py_classifier_.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//////////////////////////////////////////
// Classifier
//////////////////////////////////////////

namespace onnx_c_ops {

/////////////////////////////////////////////
// Regressor
/////////////////////////////////////////////

template <typename NTYPE>
class RuntimeTreeEnsembleRegressor : public RuntimeTreeEnsembleCommon<NTYPE> {
public:
  RuntimeTreeEnsembleRegressor() : RuntimeTreeEnsembleCommon<NTYPE>() {}
};

class RuntimeTreeEnsembleRegressorFloat
    : public RuntimeTreeEnsembleRegressor<float> {
public:
  RuntimeTreeEnsembleRegressorFloat() : RuntimeTreeEnsembleRegressor<float>() {}
};

class RuntimeTreeEnsembleRegressorDouble
    : public RuntimeTreeEnsembleRegressor<double> {
public:
  RuntimeTreeEnsembleRegressorDouble()
      : RuntimeTreeEnsembleRegressor<double>() {}
};

class RuntimeTreeEnsembleClassifierFloat
    : public RuntimeTreeEnsembleClassifier<float> {
public:
  RuntimeTreeEnsembleClassifierFloat()
      : RuntimeTreeEnsembleClassifier<float>() {}
};

class RuntimeTreeEnsembleClassifierDouble
    : public RuntimeTreeEnsembleClassifier<double> {
public:
  RuntimeTreeEnsembleClassifierDouble()
      : RuntimeTreeEnsembleClassifier<double>() {}
};

} // namespace onnx_c_ops

using namespace onnx_c_ops;

PYBIND11_MODULE(c_op_tree_ensemble_py_, m) {
  m.doc() =
#if defined(__APPLE__)
      "Implements runtime for operator TreeEnsembleClassifier and "
      "TreeEnsembleClassifier."
#else
      R"pbdoc(Implements runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc
<https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
and `tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/
onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_ in :epkg:`onnxruntime`.)pbdoc"
#endif
      ;

  /////////////
  // Regressor
  /////////////

  py::class_<RuntimeTreeEnsembleRegressorFloat> rgf(
      m, "RuntimeTreeEnsembleRegressorFloat",
      R"pbdoc(Implements float runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports float only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
)pbdoc");

  rgf.def(py::init<>());
  rgf.def("init", &RuntimeTreeEnsembleRegressorFloat::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  rgf.def("set", &RuntimeTreeEnsembleRegressorFloat::set,
          "Updates parallelization parameters.");
  rgf.def("compute", &RuntimeTreeEnsembleRegressorFloat::compute,
          "Computes the predictions for the random forest.");
  rgf.def("omp_get_max_threads",
          &RuntimeTreeEnsembleRegressorFloat::omp_get_max_threads,
          "Returns omp_get_max_threads from openmp library.");
  rgf.def("__sizeof__", &RuntimeTreeEnsembleRegressorFloat::get_sizeof,
          "Returns the size of the object.");

  py::class_<RuntimeTreeEnsembleRegressorDouble> rgd(
      m, "RuntimeTreeEnsembleRegressorDouble",
      R"pbdoc(Implements double runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports double only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
)pbdoc");

  rgd.def(py::init<>());
  rgd.def("init", &RuntimeTreeEnsembleRegressorDouble::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  rgd.def("set", &RuntimeTreeEnsembleRegressorFloat::set,
          "Updates parallelization parameters.");
  rgd.def("compute", &RuntimeTreeEnsembleRegressorDouble::compute,
          "Computes the predictions for the random forest.");
  rgd.def("omp_get_max_threads",
          &RuntimeTreeEnsembleRegressorDouble::omp_get_max_threads,
          "Returns omp_get_max_threads from openmp library.");
  rgd.def("__sizeof__", &RuntimeTreeEnsembleRegressorDouble::get_sizeof,
          "Returns the size of the object.");

  /////////////
  // Classifier
  /////////////

  py::class_<RuntimeTreeEnsembleClassifierFloat> clf(
      m, "RuntimeTreeEnsembleClassifierFloat",
      R"pbdoc(Implements float runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
in :epkg:`onnxruntime`. Supports float only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
)pbdoc");

  clf.def(py::init<>());
  clf.def("init", &RuntimeTreeEnsembleClassifierFloat::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  clf.def("set", &RuntimeTreeEnsembleRegressorFloat::set,
          "Updates parallelization parameters.");
  clf.def("compute", &RuntimeTreeEnsembleClassifierFloat::compute,
          "Computes the predictions for the random forest.");
  clf.def("omp_get_max_threads",
          &RuntimeTreeEnsembleClassifierFloat::omp_get_max_threads,
          "Returns omp_get_max_threads from openmp library.");
  clf.def("__sizeof__", &RuntimeTreeEnsembleClassifierFloat::get_sizeof,
          "Returns the size of the object.");

  py::class_<RuntimeTreeEnsembleClassifierDouble> cld(
      m, "RuntimeTreeEnsembleClassifierDouble",
      R"pbdoc(Implements double runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
in :epkg:`onnxruntime`. Supports double only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
)pbdoc");

  cld.def(py::init<>());
  cld.def("init", &RuntimeTreeEnsembleClassifierDouble::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  cld.def("set", &RuntimeTreeEnsembleRegressorFloat::set,
          "Updates parallelization parameters.");
  cld.def("compute", &RuntimeTreeEnsembleClassifierDouble::compute,
          "Computes the predictions for the random forest.");
  cld.def("omp_get_max_threads",
          &RuntimeTreeEnsembleClassifierDouble::omp_get_max_threads,
          "Returns omp_get_max_threads from openmp library.");
  cld.def("__sizeof__", &RuntimeTreeEnsembleClassifierDouble::get_sizeof,
          "Returns the size of the object.");
}
