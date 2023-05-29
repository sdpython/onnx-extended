cmake_minimum_required(VERSION 3.24.0)
project(onnx_extended VERSION 0.2.0)

#
# initialisation
#

message(STATUS "-------------------")
message(STATUS "CMAKE_VERSION=${CMAKE_VERSION}")
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
message(STATUS "CUDA_VERSION=${CUDA_VERSION}")

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/externals")

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

include("load_externals.cmake")

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
# modules
#

set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

include("targets/_validation.cmake")
include("targets/ortinf.cmake")
include("targets/c_op_conv_.cmake")
include("targets/c_op_tree_ensemble_py_.cmake")
include("targets/cuda_example_py.cmake")
include("targets/vector_function_cy.cmake")

set(ORTOPS_INCLUDE_DIR "${ROOT_INCLUDE_PATH}/onnx_extended/ortops")
include("targets/ortops_tutorial_cpu.cmake")
include("targets/ortops_tutorial_cuda.cmake")

#
# write version
#

file(WRITE "../onnx_extended/_config.py" "${config_content}")

#
# test
#

include(CTest)
enable_testing()

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
