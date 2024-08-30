// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#define py_array_ntype py::array_t<NTYPE, py::array::c_style | py::array::forcecast>

#define py_array_int64 py::array_t<int64_t, py::array::c_style | py::array::forcecast>

#include "cpu/c_op_svm_common_.hpp"
#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace onnx_c_ops {

template <typename NTYPE> class RuntimeSVMRegressor : public RuntimeSVMCommon<NTYPE> {
public:
  RuntimeSVMRegressor() : RuntimeSVMCommon<NTYPE>() {}
  ~RuntimeSVMRegressor() {}

  void init(py_array_ntype coefficients, py_array_ntype kernel_params,
            const std::string &kernel_type, int64_t n_supports, int64_t one_class,
            const std::string &post_transform, py_array_ntype rho,
            py_array_ntype support_vectors) {
    std::vector<NTYPE> vcoefficients, vkernel_params, vrho, vsupport_vectors;
    array2vector(vcoefficients, coefficients, NTYPE);
    array2vector(vkernel_params, kernel_params, NTYPE);
    array2vector(vrho, rho, NTYPE);
    array2vector(vsupport_vectors, support_vectors, NTYPE);

    std::vector<NTYPE> empty;
    std::vector<int64_t> emptyi;
    RuntimeSVMCommon<NTYPE>::init(vcoefficients, vkernel_params, kernel_type, post_transform,
                                  vrho, vsupport_vectors, n_supports, one_class, empty, empty,
                                  emptyi, emptyi);
  }

  py::array_t<NTYPE> compute(py_array_ntype X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
      throw std::invalid_argument("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];

    py_array_ntype Z(x_dims[0]);
    {
      py::gil_scoped_release release;
      compute_gil_free(x_dims, N, stride, X, Z);
    }
    return Z;
  }

private:
  void compute_gil_free(const std::vector<int64_t> &x_dims, int64_t N, int64_t stride,
                        const py_array_ntype &X, py_array_ntype &Z) const {
    RuntimeSVMCommon<NTYPE>::compute_regressor(x_dims, N, stride, (const NTYPE *)X.data(),
                                               (NTYPE *)Z.data());
  }
};

template <typename NTYPE> class RuntimeSVMClassifier : public RuntimeSVMCommon<NTYPE> {
public:
  RuntimeSVMClassifier() : RuntimeSVMCommon<NTYPE>() {}
  ~RuntimeSVMClassifier() {}

  void init(py_array_int64 classlabels_int64s,
            const std::vector<std::string> &classlabels_strings, py_array_ntype coefficients,
            py_array_ntype kernel_params, const std::string &kernel_type,
            const std::string &post_transform, py_array_ntype prob_a, py_array_ntype prob_b,
            py_array_ntype rho, py_array_ntype support_vectors,
            py_array_int64 vectors_per_class) {
    std::vector<NTYPE> proba, probb, sv, vrho, vcoef, kp;
    std::vector<int64_t> v_per_class, classlabels_ints;
    array2vector(proba, prob_a, NTYPE);
    array2vector(probb, prob_b, NTYPE);
    array2vector(sv, support_vectors, NTYPE);
    array2vector(proba, prob_a, NTYPE);
    array2vector(vrho, rho, NTYPE);
    array2vector(vcoef, coefficients, NTYPE);
    array2vector(kp, kernel_params, NTYPE);
    array2vector(v_per_class, vectors_per_class, int64_t);
    if (classlabels_strings.size() > 0)
      throw std::invalid_argument("This runtime only handles integers.");
    array2vector(classlabels_ints, classlabels_int64s, int64_t);

    RuntimeSVMCommon<NTYPE>::init(vcoef, kp, kernel_type, post_transform, vrho, sv, 0, 0, proba,
                                  probb, classlabels_ints, v_per_class);
  }

  py::tuple compute(py_array_ntype X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
      throw std::invalid_argument("X must have 2 dimensions.");
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
    int64_t n_columns = this->get_n_columns();
    std::vector<int64_t> dims{N, n_columns};

    py_array_int64 Y(N);             // one target only
    py_array_ntype Z(N * n_columns); // one target only
    {
      py::gil_scoped_release release;
      compute_gil_free(x_dims, N, stride, X, Y, Z, n_columns);
    }
    return py::make_tuple(Y, Z);
  }

private:
  void compute_gil_free(const std::vector<int64_t> &x_dims, int64_t N, int64_t stride,
                        const py_array_ntype &X, py_array_int64 &Y, py_array_ntype &Z,
                        int64_t n_columns) const {
    RuntimeSVMCommon<NTYPE>::compute_classifier(x_dims, N, stride, (const NTYPE *)X.data(),
                                                (int64_t *)Y.data(), (NTYPE *)Z.data(),
                                                n_columns);
  }
};

} // namespace onnx_c_ops

using namespace onnx_c_ops;

PYBIND11_MODULE(c_op_svm_py_, m) {
  m.doc() =
#if defined(__APPLE__)
      "Implements runtime for operator SVMRegressor and SVMClassifier."
#else
      R"pbdoc(Implements runtime for operator SVMRegressor and SVMClassifier.
The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
#endif
      ;

  py::class_<RuntimeSVMRegressor<float>> clr(
      m, "RuntimeSVMRegressorFloat",
      R"pbdoc(Implements float runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.
)pbdoc");

  clr.def(py::init<>());
  clr.def("init", &RuntimeSVMRegressor<float>::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  clr.def("compute", &RuntimeSVMRegressor<float>::compute,
          "Computes the predictions for the SVM regressor.");

  py::class_<RuntimeSVMRegressor<double>> clrd(
      m, "RuntimeSVMRegressorDouble",
      R"pbdoc(Implements Double runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.
)pbdoc");

  clrd.def(py::init<>());
  clrd.def("init", &RuntimeSVMRegressor<double>::init,
           "Initializes the runtime with the ONNX attributes in alphabetical "
           "order.");
  clrd.def("compute", &RuntimeSVMRegressor<double>::compute,
           "Computes the predictions for the SVM regressor.");

  py::class_<RuntimeSVMClassifier<float>> clf(
      m, "RuntimeSVMClassifierFloat",
      R"pbdoc(Implements float runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.
)pbdoc");

  clf.def(py::init<>());
  clf.def("init", &RuntimeSVMClassifier<float>::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  clf.def("compute", &RuntimeSVMClassifier<float>::compute,
          "Computes the predictions for the SVM regressor.");

  py::class_<RuntimeSVMClassifier<double>> cld(
      m, "RuntimeSVMClassifierDouble",
      R"pbdoc(Implements Double runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.
)pbdoc");

  cld.def(py::init<>());
  cld.def("init", &RuntimeSVMClassifier<double>::init,
          "Initializes the runtime with the ONNX attributes in alphabetical "
          "order.");
  cld.def("compute", &RuntimeSVMClassifier<double>::compute,
          "Computes the predictions for the SVM regressor.");
}
