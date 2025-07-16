#
# module: onnx_extended.onnx2.cpu._onnx2py
#
message(STATUS "+ PYBIND11 onnx_extended.onnx2.cpu._onnx2py")

add_library(lib_onnx2_cpp STATIC
  ../onnx_extended/onnx2/cpu/onnx2.cpp
  ../onnx_extended/onnx2/cpu/stream.cpp)
target_compile_definitions(lib_onnx2_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(lib_onnx2_cpp PRIVATE "${ROOT_INCLUDE_PATH}")
set_property(TARGET lib_onnx2_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

local_pybind11_add_module(
  _onnx2py OpenMP::OpenMP_CXX
  ../onnx_extended/onnx2/cpu/_onnx2py.cpp)
message(STATUS "    LINK _onnx2py <- lib_onnx2_cpp")
target_include_directories(_onnx2py PRIVATE "${ROOT_INCLUDE_PATH}")
target_link_libraries(_onnx2py PRIVATE lib_onnx2_cpp common)
