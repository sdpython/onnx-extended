#include "cuda_runtime.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

#define py_array_uint8_t py::array_t<uint8_t, py::array::c_style | py::array::forcecast>

PYBIND11_MODULE(cuda_monitor, m) {
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
}
