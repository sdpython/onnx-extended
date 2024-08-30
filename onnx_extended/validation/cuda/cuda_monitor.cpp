#include <nvml.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define py_array_float py::array_t<float, py::array::c_style | py::array::forcecast>

#define py_array_uint8_t py::array_t<uint8_t, py::array::c_style | py::array::forcecast>

std::string nvml_error_message(nvmlReturn_t code) {
  switch (code) {
  case NVML_SUCCESS:
    return "NVML_SUCCESS";
  case NVML_ERROR_UNINITIALIZED:
    return "NVML_ERROR_UNINITIALIZED";
  case NVML_ERROR_INVALID_ARGUMENT:
    return "NVML_ERROR_INVALID_ARGUMENT";
  case NVML_ERROR_INSUFFICIENT_POWER:
    return "NVML_ERROR_INSUFFICIENT_POWER";
  case NVML_ERROR_NO_PERMISSION:
    return "NVML_ERROR_NO_PERMISSION";
  case NVML_ERROR_IRQ_ISSUE:
    return "NVML_ERROR_IRQ_ISSUE";
  case NVML_ERROR_GPU_IS_LOST:
    return "NVML_ERROR_GPU_IS_LOST";
  case NVML_ERROR_UNKNOWN:
    return "NVML_ERROR_UNKNOWN";
  default:
    return "Unknown nvml error";
  }
}

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
      "nvml_init",
      []() {
        nvmlReturn_t result = nvmlInit_v2();
        if (result != NVML_SUCCESS) {
          throw std::runtime_error(std::string("nvmlInit failed: ") +
                                   nvml_error_message(result));
        }
      },
      "Initializes memory managment from nvml library.");

  m.def(
      "nvml_device_get_count",
      []() -> unsigned int {
        unsigned int count;
        nvmlReturn_t result = nvmlDeviceGetCount(&count);
        if (result != NVML_SUCCESS) {
          throw std::runtime_error(std::string("nvmlDeviceGetCount failed: ") +
                                   nvml_error_message(result));
        }
        return count;
      },
      "Returns the number of GPU units.");

  m.def(
      "nvml_shutdown",
      []() {
        nvmlReturn_t result = nvmlShutdown();
        if (result != NVML_SUCCESS) {
          throw std::runtime_error(std::string("nvmlShutdown failed: ") +
                                   nvml_error_message(result));
        }
      },
      "Closes memory managment from nvml library.");

  m.def(
      "nvml_device_get_memory_info",
      [](unsigned int device_id) -> py::tuple {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(device_id, &device);
        if (result != NVML_SUCCESS) {
          throw std::runtime_error(std::string("nvmlDeviceGetHandleByIndex_v2  failed: ") +
                                   nvml_error_message(result));
        }
        nvmlMemory_t memoryInfo;
        result = nvmlDeviceGetMemoryInfo(device, &memoryInfo);
        if (result != NVML_SUCCESS) {
          throw std::runtime_error(std::string("nvmlDeviceGetMemoryInfo failed: ") +
                                   nvml_error_message(result));
        }
        return py::make_tuple(memoryInfo.free, memoryInfo.used, memoryInfo.total);
      },
      py::arg("device") = 0,
      "Returns the free memory, the used memory, the total memory for a GPU device.");
}
