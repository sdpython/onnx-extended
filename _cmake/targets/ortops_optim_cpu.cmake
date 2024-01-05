#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ KERNEL onnx_extended.ortops.optim.cpu")

ort_add_custom_op(
  ortops_optim_cpu
  "CPU"
  onnx_extended/ortops/optim/cpu
  ../onnx_extended/ortops/optim/cpu/ort_optim_cpu_lib.cc)

target_include_directories(ortops_optim_cpu PRIVATE ${ROOT_INCLUDE_PATH})

target_include_directories(
  ortops_optim_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTAPI_INCLUDE_DIR}"
  "${ORTOPS_INCLUDE_DIR}")

target_link_libraries(
  ortops_optim_cpu
  PRIVATE
  OpenMP::OpenMP_CXX
  common_kernels
  common)

add_executable(test_optops_inference_cpp ../_unittests/ut_ortops/test_inference_tree.cpp)
target_compile_definitions(
  test_optops_inference_cpp
  PRIVATE
  PYTHON_MANYLINUX=${PYTHON_MANYLINUX}
  TESTED_CUSTOM_OPS_DLL="$<TARGET_FILE:ortops_optim_cpu>")
target_include_directories(
  test_optops_inference_cpp
  PRIVATE
  ${ROOT_UNITTEST_PATH}
  ${ROOT_PROJECT_PATH}
  ${ROOT_INCLUDE_PATH}
  ${ORT_DIR}/include)
message(STATUS "    LINK test_optops_inference_cpp <- lib_ortapi onnxruntime")
target_link_directories(test_optops_inference_cpp PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(
  test_optops_inference_cpp
  PRIVATE
  lib_ortapi
  onnxruntime
  common_kernels)
ort_add_dependency(test_optops_inference_cpp "")
add_test(NAME test_optops_inference_cpp COMMAND test_optops_inference_cpp)
