#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ onnx_extended.reference.c_ops.cpu.c_op_conv_")
message(STATUS "****** ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")

local_pybind11_add_module(
  c_op_conv_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp
  ../onnx_extended/reference/c_ops/cpu/c_op_conv_.cpp)
eigen_add_dependency(c_op_conv_)

add_executable(test_c_op_conv_cpp
               ../_unittests/ut_reference/test_c_op_conv.cpp
               ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp)
target_include_directories(test_c_op_conv_cpp PRIVATE ${ROOT_INCLUDE_PATH})
eigen_add_dependency(test_c_op_conv_cpp)
add_test(NAME test_c_op_conv_cpp COMMAND test_c_op_conv_cpp)
