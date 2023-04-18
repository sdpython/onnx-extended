#include <cuda_runtime.h>
#include "cuda_example.h"
#include "cuda_example.cuh"
#include "cuda_utils.h"

namespace cuda_example {

float vector_sum(size_t size, const float* ptr) {
  // copy memory from CPU memory to CUDA memory
  float *gpu_ptr;
  checkCudaErrors(cudaMalloc(&gpu_ptr, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr, ptr, size * sizeof(float), cudaMemcpyHostToDevice));

  // execute the code
  float result = kernel_vector_sum_reduce(gpu_ptr, size);

  // no need to copy the result back from CUDA memory to CPU memory.
  // checkCudaErrors(cudaMemcpy(ptr, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(gpu_ptr));
  return result;
}

}
