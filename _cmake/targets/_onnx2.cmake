#
# module: onnx_extended.onnx2.cpu._onnx2py
#
message(STATUS "+ PYBIND11 onnx_extended.onnx2.cpu._onnx2py")

file(GLOB_RECURSE ONNX2_SOURCES
    "../onnx_extended/onnx2/cpu/onnx*.cpp"
    "../onnx_extended/onnx2/cpu/s*.cpp"
    "../onnx_extended/onnx2/cpu/t*.cpp"
    "../onnx_extended/onnx2/cpu/*.cc"
    "../onnx_extended/onnx2/cpu/*.hpp"
    "../onnx_extended/onnx2/cpu/*.h")
add_library(lib_onnx2_cpp STATIC ${ONNX2_SOURCES})
target_compile_definitions(lib_onnx2_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(lib_onnx2_cpp PRIVATE "${ROOT_INCLUDE_PATH}")
set_property(TARGET lib_onnx2_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

local_pybind11_add_module(
  _onnx2py OpenMP::OpenMP_CXX
  ../onnx_extended/onnx2/cpu/_onnx2py.cpp)
message(STATUS "    LINK _onnx2py <- lib_onnx2_cpp")
target_include_directories(_onnx2py PRIVATE "${ROOT_INCLUDE_PATH}")
target_link_libraries(_onnx2py PRIVATE lib_onnx2_cpp common)

file(GLOB_RECURSE ONNX2_TESTS "../_unittests/ut_onnx2/*.cpp")
add_executable(test_onnx2_cpp ${ONNX2_TESTS})
target_compile_definitions(test_onnx2_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(
  test_onnx2_cpp
  PRIVATE
  "${ROOT_PROJECT_PATH}"
  "${ROOT_INCLUDE_PATH}"
  "${ROOT_UNITTEST_PATH}")
message(STATUS "    LINK test_onnx2_cpp <- lib_onnx2_cpp")
target_link_libraries(
  test_onnx2_cpp
  PRIVATE
  lib_onnx2_cpp
  common
  gtest_main)


gtest_discover_tests(test_onnx2_cpp)
