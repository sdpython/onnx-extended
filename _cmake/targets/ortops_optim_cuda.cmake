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
    ../onnx_extended/ortops/optim/cuda/addaddmulmul.cu
    ../onnx_extended/ortops/optim/cuda/addaddaddmulmulmul.cu
    ../onnx_extended/ortops/optim/cuda/addmul.cu
    ../onnx_extended/ortops/optim/cuda/add_or_mul_shared_input.cu
    ../onnx_extended/ortops/optim/cuda/mul_sigmoid.cu
    ../onnx_extended/ortops/optim/cuda/mul_mul_sigmoid.cu
    ../onnx_extended/ortops/optim/cuda/negxplus1.cu
    ../onnx_extended/ortops/optim/cuda/replace_zero.cu
    ../onnx_extended/ortops/optim/cuda/rotary.cu
    ../onnx_extended/ortops/optim/cuda/scatter_nd_of_shape.cu
    ../onnx_extended/ortops/optim/cuda/scatter_nd_of_shape_masked.cu
    ../onnx_extended/ortops/optim/cuda/submul.cu
    ../onnx_extended/ortops/optim/cuda/transpose_cast_2d.cu
    ../onnx_extended/ortops/optim/cuda/tri_matrix.cu
    ../onnx_extended/ortops/optim/cuda/ort_optim_cuda_lib.cc)

  # needed to include onnx_extended_helpers.h
  target_include_directories(
    ortops_optim_cuda
    PRIVATE
    "${ROOT_INCLUDE_PATH}"
    "${ORTAPI_INCLUDE_DIR}"
    "${ORTOPS_INCLUDE_DIR}")

endif()
