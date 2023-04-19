#
# initialization
#
# defines cuda_pybind11_add_module
#
# cuda
#

find_package(CUDA)
include(FindPackageHandleStandardArgs)

if(CUDA_FOUND)

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
  local_pybind11_add_module(${name} "" ${pybindfile})
  target_link_libraries(${name} PRIVATE ${cuda_name} stdc++)
endfunction()
