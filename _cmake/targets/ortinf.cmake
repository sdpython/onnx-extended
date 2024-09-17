#
# module: onnx_extended.ortcy.wrap.ortapi
#
message(STATUS "+ CYTHON onnx_extended.ortcy.wrap.ortapi")

add_library(lib_ortapi STATIC ../onnx_extended/ortcy/wrap/ortapi.cpp)
target_compile_definitions(lib_ortapi PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(
  lib_ortapi PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ROOT_INCLUDE_PATH})
target_link_libraries(lib_ortapi PRIVATE common)

set(ORTAPI_INCLUDE_DIR "${ROOT_PROJECT_PATH}/onnx_extended/ortcy/wrap")

cython_add_module(
  ortinf
  ../onnx_extended/ortcy/wrap/ortinf.pyx
  OpenMP::OpenMP_CXX)

message(STATUS "    LINK ortinf <- lib_ortapi onnxruntime ${ORTAPI_INCLUDE_DIR}")

ort_add_dependency(
  ortinf
  onnx_extended/ortcy/wrap)

# If ONNXRUNTIME_LIB_DIR is used, then it seems a local installation does
# does not the binaries anymore if they are removed.
target_link_directories(ortinf PRIVATE ${ORTAPI_INCLUDE_DIR})

target_link_libraries(
  ortinf
  PRIVATE
  lib_ortapi
  onnxruntime
  common_kernels)
target_include_directories(ortinf PRIVATE ${ROOT_INCLUDE_PATH})

add_executable(test_ortcy_inference_cpp ../_unittests/ut_ortcy/test_inference.cpp)
target_compile_definitions(test_ortcy_inference_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(
  test_ortcy_inference_cpp
  PRIVATE
  ${ROOT_UNITTEST_PATH}
  ${ROOT_PROJECT_PATH}
  ${ROOT_INCLUDE_PATH}
  ${ORT_DIR}/include)
message(STATUS "    LINK test_ortcy_inference_cpp <- lib_ortapi onnxruntime")
target_link_directories(test_ortcy_inference_cpp PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(
  test_ortcy_inference_cpp
  PRIVATE
  lib_ortapi
  onnxruntime
  common_kernels)
ort_add_dependency(test_ortcy_inference_cpp "")
add_test(NAME test_ortcy_inference_cpp COMMAND test_ortcy_inference_cpp)
