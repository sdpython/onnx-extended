#include "cuda_example.cuh"
#include "cuda_nvtx.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu

namespace cuda_example {

__global__ void block_vector_add(const float *a, const float *b, float *c,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void kernel_vector_add(unsigned int size, const float *gpu_ptr1,
                       const float *gpu_ptr2, float *gpu_res) {
  constexpr int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  block_vector_add<<<numBlocks, blockSize>>>(gpu_ptr1, gpu_ptr2, gpu_res, size);
}

void vector_add(unsigned int size, const float *ptr1, const float *ptr2,
                float *br, int cudaDevice) {
  // copy memory from CPU memory to CUDA memory
  NVTX_SCOPE("vector_add")
  checkCudaErrors(cudaSetDevice(cudaDevice));
  float *gpu_ptr1, *gpu_ptr2, *gpu_res;
  checkCudaErrors(cudaMalloc(&gpu_ptr1, size * sizeof(float)));
  checkCudaErrors(
      cudaMemcpy(gpu_ptr1, ptr1, size * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_ptr2, size * sizeof(float)));
  checkCudaErrors(
      cudaMemcpy(gpu_ptr2, ptr2, size * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_res, size * sizeof(float)));

  // execute the code
  kernel_vector_add(size, gpu_ptr1, gpu_ptr2, gpu_res);

  checkCudaErrors(
      cudaMemcpy(br, gpu_res, size * sizeof(float), cudaMemcpyDeviceToHost));

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr1));
  checkCudaErrors(cudaFree(gpu_ptr2));
  checkCudaErrors(cudaFree(gpu_res));
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

__global__ void kernel_sum_reduce0(float *g_idata, float *g_odata,
                                   unsigned int n) {
  extern __shared__ float sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // modulo arithmetic is slow!
    if ((tid % (2 * s)) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

float kernel_vector_sum_reduce0(float *gpu_ptr, unsigned int size,
                                int maxThreads) {
  int threads = (size < maxThreads) ? nextPow2(size) : maxThreads;
  int blocks = (size + threads - 1) / threads;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  float *gpu_block_ptr;
  checkCudaErrors(cudaMalloc(&gpu_block_ptr, blocks * sizeof(float)));
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
  kernel_sum_reduce0<<<dimGrid, dimBlock, smemSize>>>(gpu_ptr, gpu_block_ptr,
                                                      size);

  // the last reduction happens on CPU, the first step is to move
  // the data from GPU to CPU.
  float *cpu_ptr = new float[blocks];
  checkCudaErrors(cudaMemcpy(cpu_ptr, gpu_block_ptr, blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));
  float gpu_result = 0;
  for (int i = 0; i < blocks; ++i) {
    gpu_result += cpu_ptr[i];
  }
  checkCudaErrors(cudaFree(gpu_block_ptr));
  delete[] cpu_ptr;
  return gpu_result;
}

float vector_sum0(unsigned int size, const float *ptr, int maxThreads,
                  int cudaDevice) {
  // copy memory from CPU memory to CUDA memory
  NVTX_SCOPE("vector_sum0")
  float *gpu_ptr;
  checkCudaErrors(cudaSetDevice(cudaDevice));
  checkCudaErrors(cudaMalloc(&gpu_ptr, size * sizeof(float)));
  checkCudaErrors(
      cudaMemcpy(gpu_ptr, ptr, size * sizeof(float), cudaMemcpyHostToDevice));

  // execute the code
  float result = kernel_vector_sum_reduce0(gpu_ptr, size, maxThreads);

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr));
  return result;
}

__global__ void vector_sum(float *input, float *output, unsigned int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float sum = 0.0f;
  for (int i = tid; i < size; i += stride) {
    sum += input[i];
  }
  atomicAdd(output, sum);
}

float vector_sum_atomic(unsigned int size, const float *ptr, int maxThreads,
                        int cudaDevice) {
  NVTX_SCOPE("vector_sum_atomic")
  float *input, *output;
  float sum = 0.0f;
  cudaMalloc(&input, size * sizeof(float));
  checkCudaErrors(
      cudaMemcpy(input, ptr, size * sizeof(float), cudaMemcpyHostToDevice));
  cudaMalloc(&output, sizeof(float));
  cudaMemcpy(output, &sum, sizeof(float), cudaMemcpyHostToDevice);
  vector_sum<<<size, maxThreads>>>(input, output, size);
  cudaMemcpy(&sum, output, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(output);
  return sum;
}

} // namespace cuda_example
