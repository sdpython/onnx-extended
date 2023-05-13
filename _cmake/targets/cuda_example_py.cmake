#
# module: onnx_extended.validation.cuda.cuda_example_py
#
message(STATUS "+ onnx_extended.validation.cuda.cuda_example_py")

if(CUDA_AVAILABLE)

  set(config_content "HAS_CUDA = 1\nCUDA_VERSION = '${CUDA_VERSION}'")
  cuda_pybind11_add_module(
    cuda_example_py
    ../onnx_extended/validation/cuda/cuda_example_py.cpp
    ../onnx_extended/validation/cuda/cuda_example.cu
    ../onnx_extended/validation/cuda/cuda_example_reduce.cu)

else()
  set(config_content "HAS_CUDA = 0")
endif()
