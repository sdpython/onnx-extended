#
# module: onnx_extended.reference.c_ops.cpu.c_op_tree_ensembe_
#
message(STATUS "+ PYBIND11 onnx_extended.reference.c_ops.cpu.c_op_tree_ensembe_")

local_pybind11_add_module(
  c_op_tree_ensemble_py_ OpenMP::OpenMP_CXX
  ../onnx_extended/reference/c_ops/cpu/c_op_common.cpp
  ../onnx_extended/reference/c_ops/cpu/c_op_tree_ensemble_py_.cpp)

target_include_directories(
  c_op_tree_ensemble_py_
  PRIVATE
  ${ROOT_INCLUDE_PATH}/onnx_extended)
