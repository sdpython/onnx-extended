#
# module: onnx_extended.validation.cython.direct_blas_lapack_cy
#
message(STATUS "+ CYTHON onnx_extended.validation.cython.direct_blas_lapack_cy")

cython_add_module(
  direct_blas_lapack_cy
  ../onnx_extended/validation/cython/direct_blas_lapack_cy.pyx
  OpenMP::OpenMP_CXX)

target_include_directories(fp8 PRIVATE ${ROOT_INCLUDE_PATH})
