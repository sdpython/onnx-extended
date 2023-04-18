#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <iostream>
#include <sstream>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace cuda_example {

void kernel_vector_add(unsigned int size, const float* gpu_ptr1, const float* gpu_ptr2, float* gpu_res);

void vector_add(size_t size, const float *ptr1, const float *ptr2,
                float *ptr3);

float kernel_vector_sum_reduce(float* d_in, unsigned int d_in_len);

float vector_sum(size_t size, const float *ptr);

} // namespace cuda_example
