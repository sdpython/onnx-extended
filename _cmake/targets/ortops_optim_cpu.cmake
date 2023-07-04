#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ KERNEL onnx_extended.ortops.optim.cpu")

ort_add_custom_op(
  ortops_optim_cpu
  "CPU"
  ../onnx_extended/ortops/optim/cpu
  ../onnx_extended/ortops/optim/cpu/tree_ensemble.cc
  ../onnx_extended/ortops/optim/cpu/ort_optim_cpu_lib.cc)

target_include_directories(
  ortops_optim_cpu
  PRIVATE
  "${ORTAPI_INCLUDE_DIR}"
  "${ORTOPS_INCLUDE_DIR}"
  "${REFOPS_INCLUDE_DIR}")
