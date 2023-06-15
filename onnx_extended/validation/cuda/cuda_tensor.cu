#include "cuda_tensor.cuh"
#include <iostream>
#include <sstream>

namespace cuda_example {

std::string to_string(int value) {
  std::ostringstream st;
  st << value;
  return std::string(st.str());
}

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
    NVTE_CHECK(false, std::string("Unkown data type ") +
                          to_string((int)element_type) +
                          std::string(" and this CUDA version ") +
                          to_string(CUDA_VERSION) + std::string("."));
  }
}

void TensorData::allocate(cudaDataType_t dtype, size_t size,
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
      std::ostringstream st;
      st << "Unable to allocate " << size << " bytes on GPU.";
      NVTE_ERROR(std::string(st.str()));
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
    NVTE_CHECK(false, std::string("Unsupported device ") +
                          to_string((int)device) +
                          std::string(" for copy_from_cpu."));
  }
}

Tensor::Tensor(const char *name, size_t size, cudaDataType_t dtype,
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

} // namespace cuda_example