#
# initialization
#
# defines cuda_pybind11_add_module
#
# Defines USE_NTVX to enable profiling with NVIDIA profiler.
# CUDA_VERSION must be defined as well.

if(CUDA_VERSION)
  find_package(CUDAToolkit ${CUDA_VERSION} EXACT)
else()
  find_package(CUDAToolkit)
endif()

if(CUDAToolkit_FOUND)

  if(USE_NVTX)
    # see https://github.com/NVIDIA/NVTX
    include(CPM.cmake)

    CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE)

    message(STATUS "CUDA NTVX_FOUND=${NTVX_FOUND}")
    set(NVTX_LINK_C "nvtx3-c")
    set(NVTX_LINK_CPP "nvtx3-cpp")
    add_compile_definitions("ENABLE_NVTX")
  else()
    set(NVTX_LINK_C "")
    set(NVTX_LINK_CPP "")
    message(STATUS "CUDA NTVX not added.")
  endif()

  execute_process(
    COMMAND nvcc --version
    OUTPUT_VARIABLE NVCC_version_output
    ERROR_VARIABLE NVCC_version_error
    RESULT_VARIABLE NVCC_version_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # If the version is not the same, something like the following can be tried:
  # export PATH=/usr/local/cuda-11-8/bin:$PATH
  if(NOT NVCC_version_output MATCHES ".*${CUDA_VERSION}.*")
    message(FATAL_ERROR "CUDA_VERSION=${CUDA_VERSION} does not match nvcc "
                        "version=${NVCC_version_output}, try\n"
                        "export PATH=/usr/local/cuda-"
                        "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/bin:$PATH")
  endif()
  set(NVCC_VERSION "${NVCC_version_output}")
  math(
    EXPR
    CUDA_VERSION_INT
    "${CUDAToolkit_VERSION_MAJOR} * 1000 + ${CUDAToolkit_VERSION_MINOR} * 10"
    OUTPUT_FORMAT DECIMAL)

  set(CUDA_VERSION ${CUDAToolkit_VERSION})
  set(CUDA_LIBRARIES CUDA::cudart_static CUDA::cuda_driver CUDA::cublas_static CUDA::cublasLt_static)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDAToolkit_FOUND CUDA_VERSION CUDA_VERSION_INT CUDA_LIBRARIES NVCC_VERSION)

else()

  set(CUDA_VERSION_INT 0)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDAToolkit_NOTFOUND CUDA_VERSION CUDA_VERSION_INT "" "")

endif()

#
#! cuda_pybind11_add_module : compile a pyx file into cpp
#
# \arg:name extension name
# \arg:pybindfile pybind11 extension
# \argn: additional c++ files to compile as the cuda extension
#
function(cuda_pybind11_add_module name pybindfile)
  set(cuda_name ${name}_${provider})
  local_pybind11_add_module(${name} "" ${pybindfile})
  target_compile_definitions(${name} PRIVATE CUDA_VERSION=${CUDA_VERSION_INT})
  target_include_directories(${name} PRIVATE ${CUDA_INCLUDE_DIRS})
  message(STATUS "    LINK ${name} <- stdc++ ${CUDA_LIBRARIES}")
  target_link_libraries(${name} PRIVATE stdc++ ${CUDA_LIBRARIES})
  if(USE_NVTX)
    message(STATUS "    LINK ${name} <- nvtx3-cpp")
    target_link_libraries(${name} PRIVATE nvtx3-cpp)
  endif()
endfunction()
