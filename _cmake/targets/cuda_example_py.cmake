#
# module: onnx_extended.validation.cuda.cuda_example_py
#
if(CUDA_AVAILABLE)

  message(STATUS "+ CYTHON CUDA onnx_extended.validation.cuda.cuda_example_py")

  cuda_pybind11_add_module(
    cuda_example_py
    ../onnx_extended/validation/cuda/cuda_example_py.cpp
    ../onnx_extended/validation/cuda/cuda_example.cu
    ../onnx_extended/validation/cuda/cuda_example_reduce.cu)

endif()
