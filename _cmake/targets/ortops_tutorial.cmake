#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ onnx_extended.ortops.tutorial")

ort_add_custom_op(
  ortops_tutorial_cpu
  ../onnx_extended/ortopt/tutorial/cpu/my_kernel.cc
  ../onnx_extended/ortopt/tutorial/cpu/ort_tutorial_cpu_lib.cc)
