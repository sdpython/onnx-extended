#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

std::tuple<double, double> evaluate_sparse(int64_t n, const float *v, int64_t n_row,
                                           int64_t n_col, int n_random, int n_number,
                                           int n_repeat, bool dense);

} // namespace validation
