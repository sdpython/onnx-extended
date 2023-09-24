#
# module: common C++ libraries
#
message(STATUS "+ KERNEL onnx_extended.common_kernels")
add_library(
    common_kernels
    STATIC
    ../onnx_extended/cpp/c_op_allocation.cpp
    ../onnx_extended/cpp/c_op_common_parameters.cpp)
target_compile_definitions(common_kernels PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(common_kernels PRIVATE "${ROOT_INCLUDE_PATH}")
