cmake_minimum_required(VERSION 3.25.0)
project(onnx_extended VERSION ${ONNX_EXTENDED_VERSION})


# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
# see https://cmake.org/cmake/help/latest/policy/CMP0104.html
# cmake_policy(SET CMP0104 OLD) # deprecated

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/externals")


#
# Packages and constants
#

include("constants.cmake")
include("load_externals.cmake")

#
# modules
#

message(STATUS "--------------------------------------------")
set(ROOT_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_extended/cpp/include)
set(ROOT_UNITTEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../_unittests)
message(STATUS "ROOT_PROJECT_PATH=${ROOT_PROJECT_PATH}")
message(STATUS "ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")
message(STATUS "ROOT_UNITTEST_PATH=${ROOT_UNITTEST_PATH}")
message(STATUS "--------------------------------------------")

#
# common libraries for everything
#

include("targets/common.cmake")

#
# standalone modules
#

include("targets/_validation.cmake")
include("targets/cuda_example_py.cmake")
include("targets/vector_function_cy.cmake")
include("targets/fp8_cy.cmake")

#
# common libraries for kernel implementations
#

include("targets/common_kernels.cmake")

#
# new kernels for the onnx python runtime
#

include("targets/ortinf.cmake")
include("targets/c_op_conv_.cmake")
include("targets/c_op_tree_ensemble_py_.cmake")

#
# new kernels for onnxruntime
#

set(ORTOPS_INCLUDE_DIR "${ROOT_INCLUDE_PATH}/onnx_extended/ortops")
set(REFOPS_INCLUDE_DIR "${ROOT_INCLUDE_PATH}/onnx_extended/reference/c_ops")

include("targets/ortops_tutorial_cpu.cmake")
include("targets/ortops_tutorial_cuda.cmake")
include("targets/ortops_optim_cpu.cmake")

#
# finalize
#

include("finalize.cmake")

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
