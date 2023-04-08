cmake_minimum_required(VERSION 3.24.0)

project(onnx_extended VERSION 0.1.0)

######
# Test
######

include(CTest)
enable_testing()

##########
# Packages
##########

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  find_package(Python REQUIRED COMPONENTS Interpreter)
else()
  find_package(Python REQUIRED COMPONENTS NumPy)
endif()
find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
if (Python_FOUND)
    message("-- Found Python ${Python_VERSION}.")
    message("-- Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
    message("-- Python Libraries: ${Python_LIBRARIES}")
else()
    message(FATAL_ERROR "Python was not found.")
endif()

find_package(Cython REQUIRED)
if(Cython_FOUND)
    message("-- Found Cython ${Cython_VERSION}")
    message("-- NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Module cython is not installed.")
endif()

find_package(LocalPyBind11 REQUIRED)
if(LocalPyBind11_FOUND)
    message("-- Found LocalPyBind11 ${pybind11_VERSION}")
else()
    message(FATAL_ERROR "Module pybind11 is not installed.")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("-- Found OpenMP ${OpenMP_VERSION}")
else()
    message(FATAL_ERROR "OpenMP cannot be found.")
endif()

###################
# Compiling options
###################

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

#####################
# pybind11 extensions
#####################

local_pybind11_add_module(
    _validation
    ../onnx_extended/validation/_validation.cpp
    ../onnx_extended/validation/speed_metrics.cpp
    ../onnx_extended/validation/vector_sum.cpp)

local_pybind11_add_module(
    c_op_conv_
    ../onnx_extended/reference/c_ops/c_op_common.cpp
    ../onnx_extended/reference/c_ops/c_op_conv_.cpp)

local_pybind11_add_module(
    c_op_tree_ensemble_py_
    ../onnx_extended/reference/c_ops/c_op_common.cpp
    ../onnx_extended/reference/c_ops/c_op_tree_ensemble_py_.cpp)

###################
# cython extensions
###################

cython_add_module(vector_function_cy ../onnx_extended/validation/vector_function_cy.pyx
    ../onnx_extended/validation/vector_function.cpp)

#####
# omp
#####

target_link_libraries(vector_function_cy PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(_validation PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(c_op_conv_ PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(c_op_tree_ensemble_py_ PUBLIC OpenMP::OpenMP_CXX)

#######
# Final
#######

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
