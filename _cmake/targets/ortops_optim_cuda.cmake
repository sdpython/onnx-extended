#
# module: onnx_extended.ortops.optim.cuda
#

if(CUDA_AVAILABLE)

  message(STATUS "+ KERNEL onnx_extended.ortops.optim.cuda")

  ort_add_custom_op(
    ortops_optim_cuda
    CUDA
    onnx_extended/ortops/optim/cuda
    ../onnx_extended/cpp/onnx_extended_helpers.cpp
    ../onnx_extended/ortops/optim/cuda/mulmul.cu
    ../onnx_extended/ortops/optim/cuda/scatter_nd_of_shape.cu
    ../onnx_extended/ortops/optim/cuda/ort_optim_cuda_lib.cc)

  # needed to include onnx_extended_helpers.h
  target_include_directories(
    ortops_optim_cuda
    PRIVATE
    "${ROOT_INCLUDE_PATH}"
    "${ORTAPI_INCLUDE_DIR}"
    "${ORTOPS_INCLUDE_DIR}")

endif()
