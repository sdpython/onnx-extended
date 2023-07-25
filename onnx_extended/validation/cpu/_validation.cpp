#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "speed_metrics.h"
#include "vector_sum.h"

namespace py = pybind11;
using namespace validation;

PYBIND11_MODULE(_validation, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations."
#else
      R"pbdoc(C++ experimental implementations.)pbdoc"
#endif
      ;

  m.def("benchmark_cache", &benchmark_cache, py::arg("size"),
        py::arg("verbose") = true,
        R"pbdoc(Runs a benchmark to measure the cache performance.
The function measures the time for N random accesses in array of size N
and returns the time divided by N.
It copies random elements taken from the array size to random
position in another of the same size. It does that *size* times
and return the average time per move.
See example :ref:`l-example-bench-cpu`.

:param size: array size
:return: average time per move
)pbdoc");

  py::class_<ElementTime> clf(m, "ElementTime");
  clf.def(py::init<int64_t, int64_t, double>());
  clf.def_readwrite("trial", &ElementTime::trial);
  clf.def_readwrite("row", &ElementTime::row);
  clf.def_readwrite("time", &ElementTime::time);

  m.def("benchmark_cache_tree", &benchmark_cache_tree,
        py::arg("n_rows") = 100000, py::arg("n_features") = 50,
        py::arg("n_trees") = 200, py::arg("tree_size") = 4096,
        py::arg("max_depth") = 10, py::arg("search_step") = 64,
        R"pbdoc(Simulates the prediction of a random forest.
Returns the time taken by every rows for a function doing
random addition between an element from the same short buffer and
another one taken from a list of trees.
See example :ref:`l-example-bench-cpu`.

:param n_rows: number of rows of the whole batch size
:param n_features: number of features
:param n_trees: number of trees
:param tree_size: size of a tree (= number of nodes * sizeof(node) / sizeof(float))
:param max_depth: depth of a tree
:param search_step: evaluate every...
:return: array of time take for every row 
)pbdoc");

  m.def("vector_sum", &vector_sum, py::arg("n_columns"), py::arg("values"),
        py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. This function is slower than
:func:`vector_sum_array <onnx_extended.validation.cpu._validation.vector_sum_array>`
as this function copies the data from an array to a `std::vector`.
This copy (and allocation) is bigger than the compution itself.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements
)pbdoc");

  m.def("vector_sum_array", &vector_sum_array, py::arg("n_columns"),
        py::arg("values"), py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements
)pbdoc");

  m.def("vector_sum_array_parallel", &vector_sum_array_parallel,
        py::arg("n_columns"), py::arg("values"), py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation is parallelized.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements
)pbdoc");

  m.def("vector_sum_array_avx", &vector_sum_array_avx, py::arg("n_columns"),
        py::arg("values"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation uses AVX instructions
(see `AVX API
<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>`_).

:param n_columns: number of columns
:param values: all values in an array
:return: sum of all elements
)pbdoc");

  m.def("vector_sum_array_avx_parallel", &vector_sum_array_avx_parallel,
        py::arg("n_columns"), py::arg("values"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation uses AVX instructions
and parallelization (see `AVX API
<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>`_).

:param n_columns: number of columns
:param values: all values in an array
:return: sum of all elements
)pbdoc");

  m.def("vector_add", &vector_add, py::arg("v1"), py::arg("v2"),
        R"pbdoc(Computes the addition of 2 vectors of any dimensions.
It assumes both vectors have the same dimensions (no broadcast).).

:param v1: first vector
:param v2: second vector
:return: new vector
)pbdoc");
}
