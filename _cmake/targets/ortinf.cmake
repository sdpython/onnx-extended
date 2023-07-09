#
# module: onnx_extended.ortcy.wrap.ortapi
#
message(STATUS "+ CYTHON onnx_extended.ortcy.wrap.ortapi")

add_library(lib_ortapi STATIC ../onnx_extended/ortcy/wrap/ortapi.cpp)
target_include_directories(
  lib_ortapi PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ROOT_INCLUDE_PATH})

cython_add_module(
  ortinf
  ../onnx_extended/ortcy/wrap/ortinf.pyx
  OpenMP::OpenMP_CXX)
target_link_directories(ortinf PRIVATE ${ONNXRUNTIME_LIB_DIR})
message(STATUS "    LINK ortinf <- lib_ortapi onnxruntime")
target_link_libraries(ortinf PRIVATE lib_ortapi onnxruntime common_kernels)
target_include_directories(ortinf PRIVATE ${ROOT_INCLUDE_PATH})
ort_add_dependency(ortinf ${CMAKE_CURRENT_SOURCE_DIR}/../onnx_extended/ortcy/wrap/)

set(ORTAPI_INCLUDE_DIR "${ROOT_INCLUDE_PATH}/onnx_extended/ortcy/wrap")

add_executable(test_ortcy_inference_cpp ../_unittests/ut_ortcy/test_inference.cpp)
target_include_directories(
  test_ortcy_inference_cpp
  PRIVATE
  ${ROOT_UNITTEST_PATH}
  ${ROOT_PROJECT_PATH}
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
