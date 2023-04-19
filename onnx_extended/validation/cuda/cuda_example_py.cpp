#include "cuda_example.cuh"
#include "cuda_example_reduce.cuh"
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

  m.def("vector_add", [](const py_array_float& v1, const py_array_float& v2,
                         int cuda_device) -> py_array_float {
      if (v1.size() != v2.size()) {
        throw std::runtime_error("Vectors v1 and v2 have different number of elements.");
      }
      auto ha1 = v1.request();
      float* ptr1 = reinterpret_cast<float*>(ha1.ptr);
      auto ha2 = v2.request();
      float* ptr2 = reinterpret_cast<float*>(ha2.ptr);

      std::vector<int64_t> shape(v1.ndim());
      for (int i = 0; i < v1.ndim(); ++i) {
        shape[i] = v1.shape(i);
      }
      py_array_float result = py::array_t<float>(shape);
      py::buffer_info br = result.request();
      
      float * pr = static_cast<float*>(br.ptr);  // pointer on result data
      if (ptr1 == nullptr || ptr2 == nullptr || pr == nullptr) {
        throw std::runtime_error("One vector is empty.");
      }
      vector_add(v1.size(), ptr1, ptr2, pr, cuda_device);
      return result;
    }, py::arg("v1"), py::arg("v2"), py::arg("cuda_device") = 0,
    R"pbdoc(Computes the additions of two vectors
of the same size with CUDA.

:param v1: array
:param v2: array
:param cuda_device: device id (if mulitple one)
:return: addition of the two arrays
)pbdoc");

  m.def("vector_sum0", [](const py_array_float& vect, int max_threads, int cuda_device) -> float {
      if (vect.size() == 0)
        return 0;
      auto ha = vect.request();
      const float* ptr = reinterpret_cast<float*>(ha.ptr);
      return vector_sum0(static_cast<unsigned int>(vect.size()), ptr, max_threads, cuda_device);
    }, py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
    R"pbdoc(Computes the sum of all coefficients with CUDA. Naive method.

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");

  m.def("vector_sum_atomic", [](const py_array_float& vect, int max_threads, int cuda_device) -> float {
      if (vect.size() == 0)
        return 0;
      auto ha = vect.request();
      const float* ptr = reinterpret_cast<float*>(ha.ptr);
      return vector_sum_atomic(static_cast<unsigned int>(vect.size()), ptr, max_threads, cuda_device);
    }, py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
    R"pbdoc(Computes the sum of all coefficients with CUDA. Uses atomicAdd

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");

  m.def("vector_sum6", [](const py_array_float& vect, int max_threads, int cuda_device) -> float {
      if (vect.size() == 0)
        return 0;
      auto ha = vect.request();
      const float* ptr = reinterpret_cast<float*>(ha.ptr);
      return vector_sum6(static_cast<unsigned int>(vect.size()), ptr, max_threads, cuda_device);
    }, py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
    R"pbdoc(Computes the sum of all coefficients with CUDA. More efficient method.

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");

}
