#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ KERNEL onnx_extended.ortops.tutorial")

ort_add_custom_op(
  ortops_tutorial_cpu
  ../onnx_extended/ortops/tutorial/cpu
  ../onnx_extended/ortops/tutorial/cpu/my_kernel.cc
  ../onnx_extended/ortops/tutorial/cpu/ort_tutorial_cpu_lib.cc)
