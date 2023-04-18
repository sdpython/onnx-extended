#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace cuda_example {

float kernel_vector_sum_reduce(float* d_in, unsigned int d_in_len);

} // namespace cuda_example
