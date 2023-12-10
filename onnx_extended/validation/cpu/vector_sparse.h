#pragma once

#include "common/sparse_tensor.h"

#include <cstddef>
#include <stdint.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

namespace py = pybind11;

namespace validation {

py_array_float sparse_struct_to_dense(const py_array_float &v);

py_array_float dense_to_sparse_struct(const py_array_float &v);

py::list sparse_struct_to_unordered_map(const py_array_float &v);

} // namespace validation
