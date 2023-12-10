#include "cuda_example.cuh"
#include "cuda_example_reduce.cuh"
#include "cuda_nvtx.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu

namespace cuda_example {

#define reduce6_block_and_sync(I, I2)                                                          \
  if ((blockSize >= I) && (tid < I2)) {                                                        \
    sdata[tid] = mySum = mySum + sdata[tid + I2];                                              \
  }                                                                                            \
  __syncthreads();

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void kernel_reduce6(const T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ T sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  // reduction per threads on all blocks
  T mySum = 0;
  while (i < n) {
    mySum += g_idata[i];

    if (nIsPow2 || i + blockSize < n) {
      mySum += g_idata[i + blockSize];
    }

    i += gridSize;
  }

  // using shared memory to store the reduction
  sdata[tid] = mySum;
  __syncthreads();

  // reduction within a block in shared memory
  reduce6_block_and_sync(512, 256);
  reduce6_block_and_sync(256, 128);
  reduce6_block_and_sync(128, 64);

#if (__CUDA_ARCH__ >= 300)
  if (tid < 32) {
    if (blockSize >= 64) {
      mySum += sdata[tid + 32];
    }
    // Reduce final warp using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      // https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
      mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
    }
  }
#else
  // fully unroll reduction within a single warp
  reduce6_block_and_sync(64, 32);
  reduce6_block_and_sync(32, 16);
  reduce6_block_and_sync(16, 8);
  reduce6_block_and_sync(8, 4);
  reduce6_block_and_sync(4, 2);
  reduce6_block_and_sync(2, 1);
#endif

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}

bool isPow2(unsigned int n) {
  if (n == 0)
    return false;
  return (n & (n - 1)) == 0;
}

#define case_vector_sum_6_block(T, I, B)                                                       \
  case I:                                                                                      \
    kernel_reduce6<T, I, B><<<dimGrid, dimBlock, smemSize>>>(gpu_ptr, gpu_block_ptr, size);    \
    break;

float kernel_vector_sum_6(unsigned int size, const float *gpu_ptr, int maxThreads) {

  int threads = (size < maxThreads) ? nextPow2(size) : maxThreads;
  int blocks = (size + threads - 1) / threads;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  float *gpu_block_ptr;
  checkCudaErrors(cudaMalloc(&gpu_block_ptr, blocks * sizeof(float)));
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size)) {
    switch (threads) {
      case_vector_sum_6_block(float, 512, true);
      case_vector_sum_6_block(float, 256, true);
      case_vector_sum_6_block(float, 128, true);
      case_vector_sum_6_block(float, 64, true);
      case_vector_sum_6_block(float, 32, true);
      case_vector_sum_6_block(float, 16, true);
      case_vector_sum_6_block(float, 8, true);
      case_vector_sum_6_block(float, 4, true);
      case_vector_sum_6_block(float, 2, true);
      case_vector_sum_6_block(float, 1, true);
    }
  } else {
    switch (threads) {
      case_vector_sum_6_block(float, 512, false);
      case_vector_sum_6_block(float, 256, false);
      case_vector_sum_6_block(float, 128, false);
      case_vector_sum_6_block(float, 64, false);
      case_vector_sum_6_block(float, 32, false);
      case_vector_sum_6_block(float, 16, false);
      case_vector_sum_6_block(float, 8, false);
      case_vector_sum_6_block(float, 4, false);
      case_vector_sum_6_block(float, 2, false);
      case_vector_sum_6_block(float, 1, false);
    }
  }

  // the last reduction happens on CPU, the first step is to move
  // the data from GPU to CPU.
  float *cpu_ptr = new float[blocks];
  checkCudaErrors(
      cudaMemcpy(cpu_ptr, gpu_block_ptr, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu_result = 0;
  for (int i = 0; i < blocks; ++i) {
    gpu_result += cpu_ptr[i];
  }
  checkCudaErrors(cudaFree(gpu_block_ptr));
  delete[] cpu_ptr;
  return gpu_result;
}

float vector_sum6(unsigned int size, const float *ptr, int maxThreads, int cudaDevice) {
  // copy memory from CPU memory to CUDA memory
  NVTX_SCOPE("vector_sum6")
  float *gpu_ptr;
  checkCudaErrors(cudaSetDevice(cudaDevice));
  checkCudaErrors(cudaMalloc(&gpu_ptr, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr, ptr, size * sizeof(float), cudaMemcpyHostToDevice));

  // execute the code
  float result = kernel_vector_sum_6(size, gpu_ptr, maxThreads);

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr));
  return result;
}

} // namespace cuda_example
