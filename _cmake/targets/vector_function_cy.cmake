#
# module: onnx_extended.validation.cython.vector_function_cy
#
message(STATUS "+ CYTHON onnx_extended.validation.cython.vector_function_cy")

cython_add_module(
  vector_function_cy
  ../onnx_extended/validation/cython/vector_function_cy.pyx
  OpenMP::OpenMP_CXX
  ../onnx_extended/validation/cpu/vector_function.cpp)
