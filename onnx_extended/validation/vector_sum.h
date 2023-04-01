#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

float vector_sum(int nc, const std::vector<float>& values, bool by_rows);

} // namespace validation
