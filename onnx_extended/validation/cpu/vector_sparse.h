#pragma once

#include "common/sparse_tensor.h"

#include <cstddef>
#include <stdint.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>
#define py_array_uint32 py::array_t<uint32_t, py::array::c_style | py::array::forcecast>

namespace py = pybind11;

namespace validation {

py::tuple sparse_struct_indices_values(const py_array_float &v);

py_array_float sparse_struct_to_dense(const py_array_float &v);

py_array_float dense_to_sparse_struct(const py_array_float &v);

py::list sparse_struct_to_maps(const py_array_float &v);

py::tuple sparse_struct_to_csr(const py_array_float &v);

std::vector<std::tuple<double, double, double>> evaluate_sparse(const float *v, int64_t n_rows,
                                                                int64_t n_cols, int random,
                                                                int ntimes, int repeat,
                                                                int test);

} // namespace validation
