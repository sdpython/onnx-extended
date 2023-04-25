cmake_minimum_required(VERSION 3.24.0)

project(onnx_extended VERSION 0.1.0)

#
# initialisation
#

message(STATUS "-------------------")
message(STATUS "USE_SETUP_PYTHON=${USE_SETUP_PYTHON}")
message(STATUS "USE_PYTHON_SETUP=${USE_PYTHON_SETUP}")
message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
message(STATUS "PYTHON_VERSION_MM=${PYTHON_VERSION_MM}")
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}")
message(STATUS "PYTHON_LIBRARY=${PYTHON_LIBRARY}")
message(STATUS "PYTHON_LIBRARY_DIR=${PYTHON_LIBRARY_DIR}")
message(STATUS "PYTHON_NUMPY_INCLUDE_DIR=${PYTHON_NUMPY_INCLUDE_DIR}")
message(STATUS "PYTHON_MODULE_EXTENSION=${PYTHON_MODULE_EXTENSION}")
message(STATUS "PYTHON_NUMPY_VERSION=${PYTHON_NUMPY_VERSION}")
message(STATUS "USE_CUDA=${USE_CUDA}")
message(STATUS "USE_NVTX=${USE_NVTX}")
message(STATUS "ENV-PATH=$ENV{PATH}")
message(STATUS "ENV-PYTHONPATH=$ENV{PYTHONPATH}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#
# C++ 17
#

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

#
# Packages
#

message(STATUS "-------------------")
find_package(MyPython)
if(NOT ${PYTHON_VERSION} MATCHES ${Python3_VERSION})
  string(LENGTH PYTHON_VERSION_MM PYTHON_VERSION_MM_LENGTH)
  string(SUBSTRING Python3_VERSION
         0 PYTHON_VERSION_MM_LENGTH
         Python3_VERSION_MM)
  if(${PYTHON_VERSION_MM} MATCHES ${Python3_VERSION_MM})
    message(WARNING
            "cmake selects a different python micr  o version "
            "${Python3_VERSION} than ${PYTHON_VERSION}.")
  else()
    message(FATAL_ERROR
            "cmake selects a different python minor version "
            "${Python3_VERSION_MM} than ${PYTHON_VERSION_MM}.")
  endif()
  # installation of cython, numpy
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install cython numpy
    OUTPUT_VARIABLE install_version_output
    ERROR_VARIABLE install_version_error
    RESULT_VARIABLE install_version_result)
  message(STATUS "install_version_output=${install_version_output}")
  message(STATUS "install_version_error=${install_version_error}")
  message(STATUS "install_version_result=${install_version_result}")
endif()
if(MyPython_FOUND)
  message(STATUS "Python3_VERSION=${Python3_VERSION}")
  message(STATUS "Python3_LIBRARY=${Python3_LIBRARY}")
  message(STATUS "Python3_LIBRARY_RELEASE=${Python3_LIBRARY_RELEASE}")
else()
  message(FATAL_ERROR "Unable to find Python through MyPython.")
endif()

message(STATUS "-------------------")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP ${OpenMP_VERSION}")
  set(OMP_INCLUDE_DIR "")
else()
  # see https://github.com/microsoft/LightGBM/blob/master/CMakeLists.txt#L148
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MAC_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_FLAGS "${MAC_FLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_FLAGS "${MAC_FLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include")
  set(OpenMP_C_LIB_NAMES omp)
  set(OpenMP_CXX_LIB_NAMES omp)
  set(OMP_INCLUDE_DIR ${HOMEBREW_LIBOMP_PREFIX}/include)
  set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "Found(2) OpenMP ${OpenMP_VERSION}")
  else()
    message(FATAL_ERROR "OpenMP cannot be found.")
  endif()
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

if(USE_CUDA)
  find_package(CudaExtension)
  if(CUDA_FOUND)
    message(STATUS "CUDA_VERSION=${CUDA_VERSION}")
    message(STATUS "CUDA_HAS_FP16=${CUDA_HAS_FP16}")
    message(STATUS "CUDA_LIBRARIES=${CUDA_LIBRARIES}")
    message(STATUS "CUDA_CUBLAS_LIBRARIES=${CUDA_CUBLAS_LIBRARIES}")
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_cudart_static_LIBRARY=${CUDA_cudart_static_LIBRARY}")
    message(STATUS "CUDA_cudadevrt_LIBRARY=${CUDA_cudadevrt_LIBRARY}")
    message(STATUS "CUDA_cupti_LIBRARY=${CUDA_cupti_LIBRARY}")
    message(STATUS "CUDA_curand_LIBRARY=${CUDA_curand_LIBRARY}")
    message(STATUS "CUDA_cusolver_LIBRARY=${CUDA_cusolver_LIBRARY}")
    message(STATUS "CUDA_cusparse_LIBRARY=${CUDA_cusparse_LIBRARY}")
    message(STATUS "CUDA_nvToolsExt_LIBRARY=${CUDA_nvToolsExt_LIBRARY}")
    message(STATUS "CUDA_OpenCL_LIBRARY=${CUDA_OpenCL_LIBRARY}")
    message(STATUS "CUDA NVTX_LINK_C=${NVTX_LINK_C}")
    message(STATUS "CUDA NVTX_LINK_CPP=${NVTX_LINK_CPP}")
    set(CUDA_AVAILABLE 1)
  else()
    message(STATUS "Module CudaExtension is not installed.")
    set(CUDA_AVAILABLE 0)
  endif()
else()
  message(STATUS "Module CudaExtension is disabled.")
  set(CUDA_AVAILABLE 0)
endif()


message(STATUS "-------------------")
find_package(Ort REQUIRED)
if(Ort_FOUND)
  message(STATUS "Ort_VERSION=${Ort_VERSION}")
  message(STATUS "ORT_URL=${ORT_URL}")
  message(STATUS "ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNXRUNTIME_LIB_DIR=${ONNXRUNTIME_LIB_DIR}")
  message(STATUS "ORT_LIB_FILES=${ORT_LIB_FILES}")
  message(STATUS "ORT_LIB_HEADER=${ORT_LIB_HEADER}")
else()
  message(FATAL_ERROR "onnxruntime is not installed.")
endif()
message(STATUS "-------------------")

#
# constant
#

set(TEST_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../_unittests")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_constants.h.in
  ${TEST_FOLDER}/test_constants.h
)

#
# Compiling options
#

# AVX instructions
if(MSVC)
  # disable warning for #pragma unroll
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  add_compile_options(/wd4068)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if(APPLE)
  set(ENV(LDFLAGS) "-L/usr/local/opt/llvm/lib")
  set(ENV(CPPFLAGS) "-I/usr/local/opt/llvm/include")
endif()

#
# static libraries
#

add_library(lib_validation_cpp STATIC
  ../onnx_extended/validation/cpu/speed_metrics.cpp
  ../onnx_extended/validation/cpu/vector_function.cpp)
set_property(TARGET lib_validation_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(lib_ortapi STATIC ../onnx_extended/ortcy/wrap/ortapi.cpp)
target_include_directories(lib_ortapi PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
#target_link_libraries(lib_ortapi PRIVATE onnxruntime)

#
# pybind11 extensions
#

local_pybind11_add_module(
  _validation OpenMP::OpenMP_CXX
  ../onnx_extended/validation/cpu/_validation.cpp
  ../onnx_extended/validation/cpu/vector_sum.cpp)
target_link_libraries(_validation PRIVATE lib_validation_cpp)

local_pybind11_add_module(
  c_op_conv_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp
  ../onnx_extended/reference/c_ops/cpu/c_op_conv_.cpp)

local_pybind11_add_module(
  c_op_tree_ensemble_py_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp
  ../onnx_extended/reference/c_ops/cpu/c_op_tree_ensemble_py_.cpp)

if(CUDA_AVAILABLE)

  set(config_content "HAS_CUDA = 1\nCUDA_VERSION = '${CUDA_VERSION}'")
  cuda_pybind11_add_module(
    cuda_example_py
    ../onnx_extended/validation/cuda/cuda_example_py.cpp
    ../onnx_extended/validation/cuda/cuda_example.cu
    ../onnx_extended/validation/cuda/cuda_example_reduce.cu)

else()
  set(config_content "HAS_CUDA = 0")
endif()

file(WRITE "../onnx_extended/_config.py" "${config_content}")

#
# cython extensions
#

cython_add_module(
  vector_function_cy
  ../onnx_extended/validation/cython/vector_function_cy.pyx
  OpenMP::OpenMP_CXX
  ../onnx_extended/validation/cpu/vector_function.cpp)

cython_add_module(
  ortinf
  ../onnx_extended/ortcy/wrap/ortinf.pyx
  OpenMP::OpenMP_CXX)
target_link_directories(ortinf PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(ortinf PRIVATE lib_ortapi onnxruntime)
ort_add_dependency(ortinf ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_extended/ortcy/wrap/)
    
#
# onnxruntime
#

#
# test
#

include(CTest)
enable_testing()
set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(test_validation_cpp ../_unittests/ut_validation/test_vector_sum.cpp)
target_include_directories(test_validation_cpp PRIVATE ${ROOT_INCLUDE_PATH})
target_link_libraries(test_validation_cpp PRIVATE lib_validation_cpp)
add_test(NAME test_validation_cpp COMMAND test_validation_cpp)

add_executable(test_ortcy_inference_cpp ../_unittests/ut_ortcy/test_inference.cpp)
target_include_directories(
  test_ortcy_inference_cpp PRIVATE
  ${ROOT_INCLUDE_PATH}
  ${ORT_DIR}/include)
target_link_directories(test_ortcy_inference_cpp PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(test_ortcy_inference_cpp PRIVATE lib_ortapi onnxruntime)
ort_add_dependency(test_ortcy_inference_cpp "")
add_test(NAME test_ortcy_inference_cpp COMMAND test_ortcy_inference_cpp)

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
message(STATUS "-------------------")
