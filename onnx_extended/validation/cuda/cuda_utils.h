#pragma once

#include <sstream>

#define checkCudaErrors(val) _check_cuda((val), #val, __FILE__, __LINE__)

template<typename T>
void _check_cuda(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::stringstream strstr;
    strstr << "CUDA error at: " << file << ":" << line << std::endl;
    strstr << cudaGetErrorString(err) << " " << func << std::endl;
    throw strstr.str();    
  }
}
