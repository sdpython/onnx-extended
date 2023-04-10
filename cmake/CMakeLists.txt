cmake_minimum_required(VERSION 3.24.0)

project(onnx_extended VERSION 0.1.0)

#
# Test
#

include(CTest)
enable_testing()

#
# Packages
#

message(STATUS "-------------------")
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}")
message(STATUS "PYTHON_LIBRARY=${PYTHON_LIBRARY}")
message(STATUS "PYTHON_NUMPY_INCLUDE_DIR=${PYTHON_NUMPY_INCLUDE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter
                                                          Development NumPy)
if(Python_FOUND)
  message(STATUS "-------------------")
  message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
  message(STATUS "Python_VERSION=${Python_VERSION}")
  message(STATUS "Python_EXECUTABLE=${Python_EXECUTABLE}")
  message(STATUS "Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
  message(STATUS "Python_LIBRARIES=${Python_LIBRARIES}")
  message(STATUS "Python_LINK_OPTIONS=${Python_LINK_OPTIONS}")
  message(STATUS "Python_NumPy_FOUND=${Python_NumPy_FOUND}")
  message(STATUS "Python_NumPy_INCLUDE_DIRS=${Python_NumPy_INCLUDE_DIRS}")
  message(STATUS "Python_NumPy_VERSION=${Python_NumPy_VERSION}")
  message(STATUS "Python_Development_FOUND=${Python_Development_FOUND}")
else()
  message(FATAL_ERROR "Python was not found.")
endif()

if(NOT ${PYTHON_VERSION} MATCHES ${Python_VERSION})
  message(FATAL_ERROR "cmake selects a different python version "
                      "${Python_VERSION} than ${PYTHON_VERSION}.")
endif()

find_package(Cython REQUIRED)
if(Cython_FOUND)
  message(STATUS "-------------------")
  message(STATUS "Cython_VERSION=${Cython_VERSION}")
  message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Module cython is not installed.")
endif()

find_package(LocalPyBind11 REQUIRED)
if(LocalPyBind11_FOUND)
  message(STATUS "-------------------")
  message(STATUS "Found LocalPyBind11, pybind11 at ${pybind11_SOURCE_DIR}")
else()
  message(FATAL_ERROR "Module pybind11 is not installed.")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "-------------------")
  message(STATUS "Found OpenMP ${OpenMP_VERSION}")
else()
  message(FATAL_ERROR "OpenMP cannot be found.")
endif()

#
# Compiling options
#

# AVX instructions
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

# disable warning for #pragma unroll
if(MSVC)
  add_compile_options(/wd4068)
endif()

#
# pybind11 extensions
#

local_pybind11_add_module(
  _validation OpenMP::OpenMP_CXX ../onnx_extended/validation/_validation.cpp
  ../onnx_extended/validation/speed_metrics.cpp
  ../onnx_extended/validation/vector_sum.cpp)

local_pybind11_add_module(
  c_op_conv_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/c_op_common.cpp
  ../onnx_extended/reference/c_ops/c_op_conv_.cpp)

local_pybind11_add_module(
  c_op_tree_ensemble_py_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/c_op_common.cpp
  ../onnx_extended/reference/c_ops/c_op_tree_ensemble_py_.cpp)

#
# cython extensions
#

cython_add_module(
  vector_function_cy ../onnx_extended/validation/vector_function_cy.pyx
  OpenMP::OpenMP_CXX ../onnx_extended/validation/vector_function.cpp)

#
# test
#

set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_executable(test_vector_sum ../_unittests/ut_validation/test_vector_sum.cpp)
target_include_directories(test_vector_sum PRIVATE ${ROOT_INCLUDE_PATH})
add_test(_validation test_vector_sum)

#
# Final
#

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#
# Final check
#

get_property(targets_list GLOBAL PROPERTY PACKAGES_FOUND)
message(STATUS "-------------------")
message(STATUS "CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")
message(STATUS "list of found packages")
foreach(target ${targets_list})
  message(STATUS "  ${target}")
endforeach()
