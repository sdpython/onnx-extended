#
# module: onnx_extended.reference.c_ops.cpu.c_op_conv_
#
message(STATUS "+ KERNEL onnx_extended.ortops.tutorial.cuda")

ort_add_custom_op(
  ortops_tutorial_cuda
  "CUDA"
  ../onnx_extended/ortops/tutorial/cuda
  ../onnx_extended/ortops/tutorial/cuda/custom_gemm.cc
  ../onnx_extended/ortops/tutorial/cuda/ort_tutorial_cuda_lib.cc)
# needed to include helpers.h
target_include_directories(
  ortops_tutorial_cuda
  PRIVATE
  "${ORTAPI_INCLUDE_DIR}/"
  "${ORTOPS_INCLUDE_DIR}/")
