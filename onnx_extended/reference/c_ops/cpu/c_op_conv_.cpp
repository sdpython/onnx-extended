#include "c_op_conv_pybind11.h"

using namespace onnx_c_ops;

PYBIND11_MODULE(c_op_conv_, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ Reference Implementation for operator Conv."
#else
      R"pbdoc(C++ Reference Implementation for operator Conv.)pbdoc"
#endif
      ;

  py::class_<ConvFloat> clf(
      m, "ConvFloat",
      R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

  clf.def(py::init<>());
  clf.def("init", &ConvFloat::init,
          "Initializes the runtime with the ONNX attributes.");
  clf.def("compute", &ConvFloat::compute,
          "Computes the output for operator Conv.");

  py::class_<ConvDouble> cld(
      m, "ConvDouble",
      R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

  cld.def(py::init<>());
  cld.def("init", &ConvDouble::init,
          "Initializes the runtime with the ONNX attributes.");
  cld.def("compute", &ConvDouble::compute,
          "Computes the output for operator Conv.");
}
