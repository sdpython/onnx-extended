#include "cuda_example.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace cuda_example {

__global__ void block_vector_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void kernel_vector_add(unsigned int size, const float* gpu_ptr1, const float* gpu_ptr2, float* gpu_res) {
  constexpr int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  block_vector_add<<<numBlocks, blockSize>>>(gpu_ptr1, gpu_ptr2, gpu_res, size);  
}

void vector_add(size_t size, const float* ptr1, const float* ptr2, float* br) {
  // copy memory from CPU memory to CUDA memory
  float *gpu_ptr1, *gpu_ptr2, *gpu_res;
  checkCudaErrors(cudaMalloc(&gpu_ptr1, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr1, ptr1, size * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_ptr2, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr2, ptr2, size * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_res, size * sizeof(float)));

  // execute the code
  kernel_vector_add(size, gpu_ptr1, gpu_ptr2, gpu_res);

  checkCudaErrors(cudaMemcpy(br, gpu_res, size * sizeof(float), cudaMemcpyDeviceToHost));

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr1));
  checkCudaErrors(cudaFree(gpu_ptr2));
  checkCudaErrors(cudaFree(gpu_res));
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void block_sum_reduce(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata[];
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;

  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    i += gridSize; 
  }
  __syncthreads();

  if (blockSize >= 512) { 
    if (tid < 256) { 
      sdata[tid] += sdata[tid + 256]; 
    }
    __syncthreads();
  }
  if (blockSize >= 256) { 
    if (tid < 128) { 
      sdata[tid] += sdata[tid + 128]; 
    }
    __syncthreads();
  }
  if (blockSize >= 128) { 
    if (tid < 64) { 
      sdata[tid] += sdata[tid + 64]; 
    } 
    __syncthreads(); 
  }
  if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

float kernel_vector_sum_reduce(float* d_in, unsigned int d_in_len) {
	float total_sum = 0;

	constexpr unsigned int block_sz = 512; // maximum number of thread
	constexpr unsigned int max_elems_per_block = block_sz * 2;
	
	unsigned int grid_sz = 0;
	if (d_in_len <= max_elems_per_block) {
		grid_sz = (unsigned int)std::ceil(float(d_in_len) / float(max_elems_per_block));
	}
	else {
		grid_sz = d_in_len / max_elems_per_block;
		if (d_in_len % max_elems_per_block != 0)
			grid_sz++;
	}

	float* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(float) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(float) * grid_sz));

	block_sum_reduce<max_elems_per_block><<<grid_sz, block_sz>>>(d_block_sums, d_in, d_in_len);

	if (grid_sz <= max_elems_per_block) {
		float* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(unsigned int)));
		block_sum_reduce<max_elems_per_block><<<1, block_sz>>>(d_total_sum, d_block_sums, grid_sz);
		//reduce4<<<1, block_sz, sizeof(unsigned int) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);
		checkCudaErrors(cudaMemcpy(&total_sum, d_total_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_total_sum));
	}
	else {
		float* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice));
		total_sum = kernel_vector_sum_reduce(d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	checkCudaErrors(cudaFree(d_block_sums));
	return total_sum;
}

float vector_sum(size_t size, const float* ptr) {
  // copy memory from CPU memory to CUDA memory
  float *gpu_ptr;
  checkCudaErrors(cudaMalloc(&gpu_ptr, size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpu_ptr, ptr, size * sizeof(float), cudaMemcpyHostToDevice));

  // execute the code
  float result = kernel_vector_sum_reduce(gpu_ptr, size);

  // no need to copy the result back from CUDA memory to CPU memory.
  // checkCudaErrors(cudaMemcpy(ptr, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr));
  return result;
}

} // namespace cuda_example
