#
# initialization
#
# defines cuda_pybind11_add_module
#
# cuda
#

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

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDA_FOUND CUDA_VERSION CUDA_LIBRARIES)

else()

  find_package_handle_standard_args(
    CudaExtension
    VERSION_VAR "0.1"
    REQUIRED_VARS CUDA_NOTFOUND CUDA_VERSION CUDA_LIBRARIES)

endif()

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
  cuda_add_library(${cuda_name} STATIC ${ARGN})
  target_include_directories(${cuda_name} PRIVATE ${CPM_PACKAGE_NVTX_SOURCE_DIR}/include)
  local_pybind11_add_module(${name} "" ${pybindfile})
  target_link_libraries(${name} PRIVATE ${cuda_name} stdc++)
  if(USE_NVTX)
    target_link_libraries(${name} PRIVATE nvtx3-cpp)
  endif()
endfunction()
