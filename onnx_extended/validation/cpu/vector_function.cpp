#include "vector_function.h"

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark

namespace validation {

float vector_sum(int nl, int nc, const float *values, int by_rows) {
  float total = 0;
  if (by_rows) {
    for (int i = 0; i < nl; ++i) {
      for (int j = 0; j < nc; ++j) {
        total += values[i * nc + j];
      }
    }
  } else {
    for (int j = 0; j < nc; ++j) {
      for (int i = 0; i < nl; ++i) {
        total += values[i * nc + j];
      }
    }
  }
  return total;
}

} // namespace validation
