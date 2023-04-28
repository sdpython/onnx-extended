#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "c_op_deform_conv.h"

namespace py = pybind11;
using namespace onnx_c_ops;

PYBIND11_MODULE(c_op_deform_conv_, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ Reference Implementation for operator DeformConv."
#else
      R"pbdoc(C++ Reference Implementation for operator DeformConv.)pbdoc"
#endif
      ;

  py::class_<DeformConvFloat> clf(
      m, "DeformConvFloat",
      R"pbdoc(Implements float runtime for operator DeformConv. The code is inspired from
`Deformable-im2col-unfold-Deformable-Convolution-V2-PyTorch
<https://github.com/Jerrypiglet/Deformable-im2col-unfold-Deformable-Convolution-V2-PyTorch/tree/master/src/cuda>`_.
Supports double only.)pbdoc");

  clf.def(py::init<>());
  clf.def("init", &DeformConvFloat::init,
          "Initializes the runtime with the ONNX attributes.");
  clf.def("compute", &DeformConvFloat::compute,
          "Computes the output for operator Conv.");

  py::class_<DeformConvDouble> cld(
      m, "DeformConvDouble",
      R"pbdoc(Implements float runtime for operator DeformConv. The code is inspired from
`Deformable-im2col-unfold-Deformable-Convolution-V2-PyTorch
<https://github.com/Jerrypiglet/Deformable-im2col-unfold-Deformable-Convolution-V2-PyTorch/tree/master/src/cuda>`_.
Supports double only.)pbdoc");

  cld.def(py::init<>());
  cld.def("init", &DeformConvDouble::init,
          "Initializes the runtime with the ONNX attributes.");
  cld.def("compute", &DeformConvDouble::compute,
          "Computes the output for operator Conv.");
}
