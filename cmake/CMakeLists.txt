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
message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}")
message(STATUS "PYTHON_LIBRARY=${PYTHON_LIBRARY}")
message(STATUS "PYTHON_NUMPY_INCLUDE_DIR=${PYTHON_NUMPY_INCLUDE_DIR}")
message(STATUS "PYTHON_MODULE_EXTENSION=${PYTHON3_MODULE_EXTENSION}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "-------------------")
find_package(Python3 ${PYTHON_VERSION} COMPONENTS
             Interpreter NumPy Development.Module
             REQUIRED)
if(Python3_Interpreter_FOUND)
  message(STATUS "Python3_Interpreter_FOUND=${Python3_Interpreter_FOUND}")
  message(STATUS "Python3_NumPy_VERSION=${Python3_NumPy_VERSION}")
  message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
  message(STATUS "Python3_VERSION=${Python3_VERSION}")
  message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
  message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
  message(STATUS "Python3_LIBRARIES=${Python3_LIBRARIES}")
  message(STATUS "Python3_LINK_OPTIONS=${Python3_LINK_OPTIONS}")
  message(STATUS "Python3_NumPy_FOUND=${Python3_NumPy_FOUND}")
  message(STATUS "Python3_NumPy_INCLUDE_DIRS=${Python3_NumPy_INCLUDE_DIRS}")
  message(STATUS "Python3_NumPy_VERSION=${Python3_NumPy_VERSION}")
  message(STATUS "Python3_Development_FOUND=${Python3_Development_FOUND}")
  message(STATUS "Python3_MODULE_EXTENSION=${Python3_MODULE_EXTENSION}")
  message(STATUS "Python3_MODULE_PREFIX=${Python3_MODULE_PREFIX}")
  message(STATUS "Python3_SOABI=${Python3_SOABI}")
  message(STATUS "Python3_SOSABI=${Python3_SOSABI}")
else()
  message(STATUS "-------------------")
  message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
  message(FATAL_ERROR "Python was not found.")
endif()

if(NOT ${PYTHON_VERSION} MATCHES ${Python3_VERSION})
  message(FATAL_ERROR "cmake selects a different python version "
                      "${Python3_VERSION} than ${PYTHON_VERSION}.")
endif()

message(STATUS "-------------------")
find_package(Cython REQUIRED)
if(Cython_FOUND)
  message(STATUS "Cython_VERSION=${Cython_VERSION}")
  message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Module cython is not installed.")
endif()

message(STATUS "-------------------")
find_package(LocalPyBind11 REQUIRED)
if(LocalPyBind11_FOUND)
  message(STATUS "Found LocalPyBind11, pybind11 at ${pybind11_SOURCE_DIR}")
else()
  message(FATAL_ERROR "Module pybind11 is not installed.")
endif()

message(STATUS "-------------------")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP ${OpenMP_VERSION}")
else()
  message(FATAL_ERROR "OpenMP cannot be found.")
endif()
message(STATUS "-------------------")

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

add_library(lib_validation_cpp STATIC
  ../onnx_extended/validation/speed_metrics.cpp
  ../onnx_extended/validation/vector_function.cpp)
set_property(TARGET lib_validation_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

local_pybind11_add_module(
  _validation OpenMP::OpenMP_CXX
  ../onnx_extended/validation/_validation.cpp
  ../onnx_extended/validation/vector_sum.cpp)
target_link_libraries(_validation PRIVATE lib_validation_cpp)

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
add_executable(test_validation_cpp
  ../_unittests/ut_validation/test_vector_sum.cpp)
target_include_directories(test_validation_cpp PRIVATE ${ROOT_INCLUDE_PATH})
target_link_libraries(test_validation_cpp PRIVATE lib_validation_cpp)
add_test(NAME test_validation_cpp COMMAND test_validation_cpp)

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
