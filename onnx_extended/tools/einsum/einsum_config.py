from typing import Any
import numpy
import onnx

DEFAULT_OPSET = min(18, onnx.defs.onnx_opset_version())
DEFAULT_IR_VERSION = 8


def guess_proto_dtype(dtype: Any) -> int:
    """
    Returns the corresponding proto type for a numpy dtype.
    """
    if dtype == numpy.float32:
        return onnx.TensorProto.FLOAT
    if dtype == numpy.float64:
        return onnx.TensorProto.DOUBLE
    if dtype == numpy.int32:
        return onnx.TensorProto.INT32
    if dtype == numpy.int64:
        return onnx.TensorProto.INT64
    raise ValueError(f"Unexpected value for dtype {dtype!r}.")
