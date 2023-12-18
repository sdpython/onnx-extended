#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

std::vector<std::tuple<double, double, double>>
evaluate_sparse(int64_t n_elements, const float *v, int64_t n_rows, int64_t n_cols, int random,
                int number, int repeat, int test);

} // namespace validation
