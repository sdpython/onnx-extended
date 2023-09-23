#
# module: onnx_extended.validation.cython.fp8
#
message(STATUS "+ CYTHON onnx_extended.validation.cython.fp8")

cython_add_module(
  fp8
  ../onnx_extended/validation/cython/fp8.pyx
  OpenMP::OpenMP_CXX)

target_include_directories(fp8 PRIVATE ${ROOT_INCLUDE_PATH})
