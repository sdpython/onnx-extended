#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

namespace py = pybind11;

namespace validation {

// vector_sum

float vector_sum(int nc, const std::vector<float> &values, bool by_rows);

float vector_sum_array(int nc, const py_array_float &values, bool by_rows);

float vector_sum_array_parallel(int nc, const py_array_float &values, bool by_rows);

float vector_sum_array_avx(int nc, const py_array_float &values_array);

float vector_sum_array_avx_parallel(int nc, const py_array_float &values_array);

// vector_add

py_array_float vector_add(const py_array_float &v1, const py_array_float &v2);

} // namespace validation
