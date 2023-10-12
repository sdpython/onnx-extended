#pragma once

#include <stdint.h>

namespace cuda_fpemu {

enum FpemuMode {
  E4M3_RNE = 1,
};

void fpemu_cuda_forward(const int size, const float *input, uint8_t *output,
                        FpemuMode mode, bool inplace, float scale,
                        bool block_norm, int block_size);

} // namespace cuda_fpemu