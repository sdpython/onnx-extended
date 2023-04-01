#include "vector_sum.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark

namespace validation {

float vector_sum(int nc, const std::vector<float>& values, bool by_rows) {
    float total = 0;
    if (by_rows) {
        /*
        for(size_t i=0; i < values.size(); ++i) {
            total += values[i];
        }
        */
        int nl = values.size() / nc;
        for(size_t i = 0; i < nl; ++i) {
            for(size_t j = 0; j < nc; ++j) {
                total += values[i * nc + j];
            }
        }
    }
    else {
        int nl = values.size() / nc;
        for(size_t j = 0; j < nc; ++j) {
            for(size_t i = 0; i < nl; ++i) {
                total += values[i * nc + j];
            }
        }
    }
    return total;
}

} // namespace validation
