#include "cuda_example.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cuda_example;

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

PYBIND11_MODULE(cuda_example_py, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations with CUDA."
#else
      R"pbdoc(C++ experimental implementations with CUDA.)pbdoc"
#endif
      ;

  m.def("vector_sum", [](const py_array_float& vect) {
      auto ha = vect.request();
      float* ptr = reinterpret_cast<float*>(ha.ptr);
      return vector_sum(vect.size(), ptr)  ;
    }, py::arg("vect"), R"pbdoc(Computes the sum of all coefficients with CUDA.

:param vect: array
:return: sum
)pbdoc");

}
