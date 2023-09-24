#include "cuda_tensor.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#include "onnx_extended_helpers.h"

namespace cuda_example {

bool is_fp8_dtype(cudaDataType_t dtype) {
  switch (dtype) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
    return true;
  case CUDA_R_8F_E5M2:
    return true;
#endif
  default:
    return false;
  }
}

int32_t type_size(cudaDataType_t element_type) {
  switch (element_type) {
  case CUDA_R_32I:
  case CUDA_R_32F:
    return 4;
  case CUDA_R_16F:
  case CUDA_R_16BF:
    return 2;
  case CUDA_R_8I:
  case CUDA_R_8U:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
  case CUDA_R_8F_E5M2:
#endif
    return 1;
  default:
    NVTE_CHECK(false, onnx_extended_helpers::MakeString(
                          "Unkown data type ", element_type,
                          " and this CUDA version ", CUDA_VERSION, "."));
  }
}

void TensorData::allocate(cudaDataType_t dtype, std::size_t size,
                          TensorDevice device) {
  this->dtype = dtype;
  this->size = size;
  this->device = device;
  switch (device) {
  case TensorDevice::CPU:
    dptr = malloc(size * type_size(dtype));
    break;
  case TensorDevice::CUDA:
    if (cudaMalloc(&dptr, size * type_size(dtype)) != cudaSuccess) {
      NVTE_ERROR(onnx_extended_helpers::MakeString("Unable to allocate ", size,
                                                   " bytes on GPU."));
    }
    break;
  }
}

void TensorData::free() {
  if (dptr != nullptr) {
    switch (device) {
    case TensorDevice::CPU:
      ::free(dptr);
      break;
    case TensorDevice::CUDA:
      NVTE_CHECK_CUDA(cudaFree(dptr));
      break;
    }
    dptr = nullptr;
  }
}

void TensorData::copy_from_cpu(void *ptr) {
  switch (device) {
  case TensorDevice::CPU:
    memcpy(dptr, ptr, type_size(dtype) * size);
    break;
  case TensorDevice::CUDA:
    NVTE_CHECK_CUDA(
        cudaMemcpy(dptr, ptr, type_size(dtype) * size, cudaMemcpyHostToDevice));
    break;
  default:
    NVTE_CHECK(false, onnx_extended_helpers::MakeString("Unsupported device ",
                                                        (int)device,
                                                        " for copy_from_cpu."));
  }
}

Tensor::Tensor(const char *name, std::size_t size, cudaDataType_t dtype,
               TensorDevice device, TensorDevice scale_device) {
  this->name = name;
  data.allocate(dtype, size, device);
  if (is_fp8_dtype(dtype)) {
    float one = 1;
    scale.allocate(CUDA_R_32F, 1, scale_device);
    scale_inv.allocate(CUDA_R_32F, 1, scale_device);
    scale.copy_from_cpu(&one);
    scale_inv.copy_from_cpu(&one);
  }
}

Tensor::~Tensor() {
  data.free();
  scale.free();
  scale_inv.free();
  amax.free();
}

__global__ void generateRandomFloat16(__half *randomFloat16, int numElements,
                                      unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] = __float2half(randValue);
  }
}

__global__ void generateRandomBFloat16(__nv_bfloat16 *randomFloat16,
                                       int numElements, unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] = __float2bfloat16(randValue);
  }
}

__global__ void generateRandomInt8x4(int *randomInt8, int numElements,
                                     unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements / 4) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    int randValue = curand_poisson(&state, 1);
    randomInt8[tid] = randValue;
  }
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

__global__ void generateRandomFloat8E4M3FN(__nv_fp8_storage_t *randomFloat16,
                                           int numElements, unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] =
        __nv_cvt_float_to_fp8(randValue, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void generateRandomFloat8E5M2(__nv_fp8_storage_t *randomFloat16,
                                         int numElements, unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] =
        __nv_cvt_float_to_fp8(randValue, __NV_SATFINITE, __NV_E5M2);
  }
}

#endif

void Tensor::rnd() {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  switch (data.dtype) {
  case CUDA_R_32F:
    curandGenerateUniform(gen, static_cast<float *>(data.dptr), data.size);
    break;
  case CUDA_R_16F: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomFloat16<<<numBlocks, blockSize>>>(
        static_cast<__half *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
  case CUDA_R_16BF: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomBFloat16<<<numBlocks, blockSize>>>(
        static_cast<__nv_bfloat16 *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
  case CUDA_R_8I: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomInt8x4<<<numBlocks, blockSize>>>(
        static_cast<int *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomFloat8E4M3FN<<<numBlocks, blockSize>>>(
        static_cast<__nv_fp8_storage_t *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
  case CUDA_R_8F_E5M2: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomFloat8E5M2<<<numBlocks, blockSize>>>(
        static_cast<__nv_fp8_storage_t *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
#endif
  default:
    NVTE_CHECK(false, onnx_extended_helpers::MakeString(
                          "Unsupported dtype ", data.dtype, " for rnd."));
  }
  curandDestroyGenerator(gen);
}

} // namespace cuda_example