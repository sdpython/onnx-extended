#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ onnx_extended.reference.c_ops.cpu.c_op_conv_")

local_pybind11_add_module(
  c_op_conv_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp
  ../onnx_extended/reference/c_ops/cpu/c_op_conv_.cpp)
eigen_add_dependency(c_op_conv_)
