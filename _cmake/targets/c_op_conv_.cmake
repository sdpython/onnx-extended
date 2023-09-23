#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ PYBIND11 onnx_extended.reference.c_ops.cpu.c_op_conv_")

local_pybind11_add_module(
  c_op_conv_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_conv_.cpp)
eigen_add_dependency(c_op_conv_)

target_link_libraries(c_op_conv_ PRIVATE common_kernels common)
target_include_directories(c_op_conv_ PRIVATE ${ROOT_INCLUDE_PATH})

add_executable(test_c_op_conv_cpp ../_unittests/ut_reference/test_c_op_conv.cpp)
target_compile_definitions(test_c_op_conv_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_link_libraries(test_c_op_conv_cpp PRIVATE common_kernels common)
target_include_directories(
  test_c_op_conv_cpp
  PRIVATE
  ${ROOT_INCLUDE_PATH}
  ${ROOT_UNITTEST_PATH})

eigen_add_dependency(test_c_op_conv_cpp)

add_test(NAME test_c_op_conv_cpp COMMAND test_c_op_conv_cpp)
