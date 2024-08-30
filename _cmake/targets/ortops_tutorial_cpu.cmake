#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ KERNEL onnx_extended.ortops.tutorial.cpu")

ort_add_custom_op(
  ortops_tutorial_cpu
  "CPU"
  onnx_extended/ortops/tutorial/cpu
  ../onnx_extended/ortops/tutorial/cpu/custom_gemm.cc
  ../onnx_extended/ortops/tutorial/cpu/custom_tree_assembly.cc
  ../onnx_extended/ortops/tutorial/cpu/dynamic_quantize_linear.cc
  ../onnx_extended/ortops/tutorial/cpu/my_kernel.cc
  ../onnx_extended/ortops/tutorial/cpu/my_kernel_attr.cc
  ../onnx_extended/ortops/tutorial/cpu/ort_tutorial_cpu_lib.cc)

# needed to include onnx_extended_helpers.h
target_include_directories(
  ortops_tutorial_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTAPI_INCLUDE_DIR}"
  "${ORTOPS_INCLUDE_DIR}")

eigen_add_dependency(ortops_tutorial_cpu)

target_link_libraries(
  ortops_tutorial_cpu
  PRIVATE
  OpenMP::OpenMP_CXX
  common_kernels
  common)
