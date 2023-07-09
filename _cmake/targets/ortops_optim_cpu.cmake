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
  common_kernels)

