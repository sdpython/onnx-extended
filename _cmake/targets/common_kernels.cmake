#
# module: common C++ libraries
#
message(STATUS "+ KERNEL onnx_extended.ortops.optim.cpu")

add_library(common_kernels STATIC ../onnx_extended/cpp/c_op_common.cpp)
target_include_directories(common_kernels PRIVATE "${ROOT_INCLUDE_PATH}")
