#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

float vector_sum(int nl, int nc, const float *values, int by_rows);

} // namespace validation
