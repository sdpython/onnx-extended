#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "speed_metrics.h"

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
}
