#include "speed_metrics.h"

PYBIND11_MODULE(_validation, m) {
    m.doc() =
#if defined(__APPLE__)
        "C++ experimental implementations."
#else
        R"pbdoc(C++ experimental implementations.)pbdoc"
#endif
        ;

    m.def("benchmark_cache", &benchmark_cache,
        py::arg("size"), py::arg("verbose") = true,
        R"pbdoc(Runs a benchmark to measure the cache performance.
It copies random elements taken from the array size to random
position in another of the same size. It does that *size* times
and return the average time per move.

:param size: array size
:return: average time per move
)pbdoc");

    py::class_<elem_time> clf (m, "elem_time");
    clf.def(py::init<int64_t, int64_t, double>());
    clf.def_readwrite("n_trial", &elem_time::n_trial);
    clf.def_readwrite("row", &elem_time::row);
    clf.def_readwrite("time", &elem_time::time);

    m.def("benchmark_cache_tree", &benchmark_cache_tree,
        py::arg("n_rows") = 100000,
        py::arg("n_features") = 50,
        py::arg("n_trees") = 200, 
        py::arg("tree_size") = 4096, 
        py::arg("max_depth") = 10, 
        py::arg("n_trials") = 2, 
        R"pbdoc(Simulates the prediction of a random forest.
Returns the time taken by every rows.

:param n_rows: number of rows of the whole batch size
:param n_features: number of features
:param n_trees: number of trees
:param tree_size: size of a tree (= number of nodes * sizeof(node) / sizeof(float))
:param max_depth: depth of a tree
:return: array of time take for every row 
)pbdoc");
}
