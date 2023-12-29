#include "cuda_fpemu.cuh"
#include "cuda_gemm.cuh"

#include "cuda_runtime.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cuda_example;
using namespace cuda_fpemu;

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

#define py_array_uint8_t py::array_t<uint8_t, py::array::c_style | py::array::forcecast>

PYBIND11_MODULE(cuda_example_py, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations with CUDA."
#else
      R"pbdoc(C++ experimental implementations with CUDA.)pbdoc"
#endif
      ;

#if defined(CUDA_VERSION)
  m.def(
      "cuda_version", []() -> int { return CUDA_VERSION; },
      "Returns the CUDA version the project was compiled with.");
#else
  m.def(
      "cuda_version", []() -> int { return 0; },
      "CUDA was not enabled during the compilation.");
#endif

  m.def(
      "cuda_device_count",
      []() -> int {
        int devices;
        auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("Unable to retrieve the number of devices ") +
                                   std::string(cudaGetErrorString(status)));
        return devices;
      },
      "Returns the number of cuda devices.");

  m.def(
      "cuda_device_memory",
      [](int device) -> py::tuple {
        cudaSetDevice(device);
        size_t free_memory, total_memory;
        auto status = cudaMemGetInfo(&free_memory, &total_memory);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("Unable to retrieve the memory for a device ") +
                                   std::string(cudaGetErrorString(status)));
        return py::make_tuple(free_memory, total_memory);
      },
      py::arg("device") = 0, "Returns the free and total memory for a particular device.");

  m.def(
      "cuda_devices_memory",
      []() -> py::list {
        int devices;
        auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("Unable to retrieve the number of devices ") +
                                   std::string(cudaGetErrorString(status)));
        py::list res;
        size_t free_memory, total_memory;
        for (int i = 0; i < devices; ++i) {
          cudaSetDevice(i);
          status = cudaMemGetInfo(&free_memory, &total_memory);
          if (status != cudaSuccess)
            throw std::runtime_error(
                std::string("Unable to retrieve the memory for a device ") +
                std::string(cudaGetErrorString(status)));
          res.append(py::make_tuple(free_memory, total_memory));
        }
        return res;
      },
      "Returns the free and total memory for all devices.");

  m.def(
      "get_device_prop",
      [](int device_id) -> py::dict {
        cudaDeviceProp prop;
        auto status = cudaGetDeviceProperties(&prop, device_id);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("Unable to retrieve the device property ") +
                                   std::string(cudaGetErrorString(status)));
        py::dict res;
        res["name"] = py::str(prop.name);
        res["totalGlobalMem"] = prop.totalGlobalMem;
        res["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
        res["computeMode"] = prop.computeMode;
        res["major"] = prop.major;
        res["minor"] = prop.minor;
        res["isMultiGpuBoard"] = prop.isMultiGpuBoard;
        res["concurrentKernels"] = prop.concurrentKernels;
        res["totalConstMem"] = prop.totalConstMem;
        res["clockRate"] = prop.clockRate;
        res["sharedMemPerBlock"] = prop.sharedMemPerBlock;
        res["multiProcessorCount"] = prop.multiProcessorCount;
        return res;
      },
      py::arg("device_id") = 0, "Returns the device properties.");

  m.def("gemm_benchmark_test", &gemm_benchmark_test, py::arg("test_id") = 0, py::arg("N") = 10,
        py::arg("m") = 16, py::arg("n") = 16, py::arg("k") = 16, py::arg("lda") = 16,
        py::arg("ldb") = 16, py::arg("ldd") = 16,
        R"pbdoc(Benchmark Gemm on CUDA
        
:param test_id: a test configuration (int)
:param N: number of repetitions
:param m: dimensions of the matrices
:param n: dimensions of the matrices
:param k: dimensions of the matrices
:param lda: leading dimension of A
:param ldb: leading dimension of B
:param ldd: leading dimension of the result
:return: metrics in a dictionary
)pbdoc");

  py::enum_<FpemuMode>(m, "FpemuMode",
                       "Available option for parameter mode in function fpemu_cuda_forward.")
      .value("E4M3_RNE", FpemuMode::E4M3_RNE)
      .export_values();

  m.def(
      "fpemu_cuda_forward",
      [](py_array_float &input, FpemuMode mode, bool inplace, float scale, bool block_norm,
         int block_size, int cuda_device) -> py_array_float {
        py::buffer_info br = input.request();
        float *ptr_in = reinterpret_cast<float *>(br.ptr);

        if (inplace) {
          fpemu_cuda_forward(input.size(), ptr_in, ptr_in, mode, inplace, scale, block_norm,
                             block_size, cuda_device);
          return input;
        } else {
          py_array_float output = py::array_t<float>({input.size()});
          py::buffer_info bro = output.request();
          float *ptr_out = reinterpret_cast<float *>(bro.ptr);
          fpemu_cuda_forward(input.size(), ptr_in, ptr_out, mode, inplace, scale, block_norm,
                             block_size, cuda_device);
          return output;
        }
      },
      py::arg("input"), py::arg("mode") = FpemuMode::E4M3_RNE, py::arg("inplace") = false,
      py::arg("scale") = 1.0, py::arg("block_norm") = false, py::arg("block_size") = 1,
      py::arg("cuda_device") = 0, R"pbdoc(Experimental

:param input: array
:param mode: which quantization type
:param inplace: modification inplace instead of a new outoput
:param scale: scale
:param block_norm: normalization accrocess blocks
:param block_size: block size
:param cuda_device: device id (if mulitple one)
:return: forward pass
      )pbdoc");

}
