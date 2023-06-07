#
# initialization
#
# defines cuda_pybind11_add_module
#
# Defines USE_NTVX to enable profiling with NVIDIA profiler.
# CUDA_VERSION must be defined as well.

find_package(CUDA)

if(CUDA_FOUND)

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
                        "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}/bin:$PATH")
  endif()
  set(NVCC_VERSION "${NVCC_version_output}")
  math(
    EXPR
    CUDA_VERSION_INT
    "${CUDA_VERSION_MAJOR} * 1000 + ${CUDA_VERSION_MINOR} * 10"
    OUTPUT_FORMAT DECIMAL)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDA_FOUND CUDA_VERSION CUDA_VERSION_INT CUDA_LIBRARIES NVCC_VERSION)

  find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  if(CUBLAS_LIBRARY)
    message(STATUS "CUBLAS found: ${CUBLAS_LIBRARY}")
  else()
    message(FATAL_ERROR "CUBLAS not found.")
  endif()
else()

  set(CUDA_VERSION_INT 0)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDA_NOTFOUND CUDA_VERSION CUDA_VERSION_INT CUDA_LIBRARIES "")

endif()

#
#! cuda_add_library_ext(name files)
#
# \arg:name extension name
# \arg:kind SHARED or STATIC
# \argn: additional c++ files to compile as the cuda extension
#
function(cuda_add_library_ext name kind)
  cuda_add_library(${name} ${kind} ${ARGN})
  target_include_directories(
    ${name}
    PRIVATE
    ${CPM_PACKAGE_NVTX_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE CUDA_VERSION=${CUDA_VERSION_INT})
endfunction()

#
#! cuda_pybind11_add_module : compile a pyx file into cpp
#
# \arg:name extension name
# \arg:pybindfile pybind11 extension
# \argn: additional c++ files to compile as the cuda extension
#
function(cuda_pybind11_add_module name pybindfile)
  set(cuda_name ${name}_cuda)
  message(STATUS "CU ${name}::${cuda_name}")
  message(STATUS "CU ${pybindfile}")
  message(STATUS "CU ${ARGN}")
  cuda_add_library_ext(${cuda_name} STATIC ${ARGN})
  local_pybind11_add_module(${name} "" ${pybindfile})
  target_compile_definitions(${name} PRIVATE CUDA_VERSION=${CUDA_VERSION_INT})
  target_include_directories(${name} PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE ${cuda_name} stdc++)
  if(USE_NVTX)
    target_link_libraries(${name} PRIVATE nvtx3-cpp)
  endif()
endfunction()
