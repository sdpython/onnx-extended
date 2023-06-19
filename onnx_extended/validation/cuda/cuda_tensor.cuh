#pragma once
#include "cuda_nvtx.cuh"
#include "cuda_utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda_example {

std::string to_string(int value);

typedef enum TensorDevice { CPU = 0, CUDA = 1 } TensorDevice;

bool is_fp8_dtype(cudaDataType_t dtype);

int32_t type_size(cudaDataType_t element_type);

inline cudaDataType_t get_cuda_dtype(cudaDataType_t dtype) { return dtype; }

struct TensorData {
  TensorDevice device;
  cudaDataType_t dtype;
  size_t size;
  void *dptr;
  inline TensorData() {
    device = TensorDevice::CPU;
    size = 0;
    dptr = nullptr;
    dtype = CUDA_R_32F;
  }
  void allocate(cudaDataType_t dtype, size_t size, TensorDevice device);
  void free();
  void copy_from_cpu(void *ptr);
};

class Tensor {
public:
  const char *name;
  TensorData data;
  TensorData scale;
  TensorData amax;
  TensorData scale_inv;

public:
  inline Tensor(const char *name) : data(), scale(), amax(), scale_inv() {
    this->name = name;
  }
  Tensor(const char *name, size_t size, cudaDataType_t dtype = CUDA_R_32F,
         TensorDevice device = TensorDevice::CUDA,
         TensorDevice scale_device = TensorDevice::CUDA);
  ~Tensor();
  void rnd();
};

} // namespace cuda_example