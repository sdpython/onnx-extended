#
# module: onnx_extended.validation.cuda.cuda_example_py
#
if(CUDA_AVAILABLE)

  message(STATUS "+ PYBIND11 CUDA onnx_extended.validation.cuda.cuda_example_py")

  cuda_pybind11_add_module(
    cuda_example_py
    ../onnx_extended/validation/cuda/cuda_example_py.cpp
    ../onnx_extended/validation/cuda/cuda_fpemu.cu
    ../onnx_extended/validation/cuda/cuda_tensor.cu
    ../onnx_extended/validation/cuda/cuda_gemm.cu)

  target_include_directories(cuda_example_py PRIVATE ${ROOT_INCLUDE_PATH})
  target_link_libraries(cuda_example_py PRIVATE common)

endif()
