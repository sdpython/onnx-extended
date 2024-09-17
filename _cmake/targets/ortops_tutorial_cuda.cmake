#
# custom ops: onnx_extended.ortops.tutorial.cuda
#

if(CUDA_AVAILABLE)

  message(STATUS "+ KERNEL onnx_extended.ortops.tutorial.cuda")

  ort_add_custom_op(
    ortops_tutorial_cuda
    CUDA
    onnx_extended/ortops/tutorial/cuda
    ../onnx_extended/cpp/onnx_extended_helpers.cpp
    ../onnx_extended/ortops/tutorial/cuda/custom_gemm.cu
    ../onnx_extended/ortops/tutorial/cuda/matx_matmul.cu
    ../onnx_extended/ortops/tutorial/cuda/ort_tutorial_cuda_lib.cc)

  # needed to include onnx_extended_helpers.h
  target_include_directories(
    ortops_tutorial_cuda
    PRIVATE
    "${ROOT_INCLUDE_PATH}"
    "${ORTAPI_INCLUDE_DIR}"
    "${ORTOPS_INCLUDE_DIR}"
    "${matx_INCLUDE_DIR}")

  target_link_libraries(ortops_tutorial_cuda PRIVATE matx::matx)

endif()
