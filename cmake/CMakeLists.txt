cmake_minimum_required(VERSION 3.24.0)

project(onnx_extended VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(CTest)
enable_testing()

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

# needed packages
find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
find_package(Cython REQUIRED)
message(${CYTHON_EXECUTABLE})

# This comes to hand if we also need to use the NumPy C API
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "Numpy headers not found.")
endif()

# pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.4
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# List of extensions to build.

add_cython_target(vector_sum_cy CXX)
add_library(vector_sum_cy MODULE 
    ../onnx_extended/validation/vector_sum_cy.pyx
    ../onnx_extended/validation/vector_sum.cpp)
#python_extension_module(vector_sum_cy)
install(TARGETS vector_sum_cy LIBRARY DESTINATION ../onnx_extended/validations)

pybind11_add_module(_validation
    ../onnx_extended/validation/_validation.cpp
    ../onnx_extended/validation/speed_metrics.cpp
    ../onnx_extended/validation/vector_sum.cpp)

pybind11_add_module(c_op_conv_
    ../onnx_extended/reference/c_ops/c_op_common.cpp
    ../onnx_extended/reference/c_ops/c_op_conv_.cpp)

pybind11_add_module(c_op_tree_ensemble_py_
    ../onnx_extended/reference/c_ops/c_op_common.cpp
    ../onnx_extended/reference/c_ops/c_op_tree_ensemble_py_.cpp)

# omp

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vector_sum_cy PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(_validation PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(c_op_conv_ PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(c_op_tree_ensemble_py_ PUBLIC OpenMP::OpenMP_CXX)
else()
    error("OpenMP cannot be found.")
endif()

# Final

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
