from typing import Union
import numpy as np
from onnx import SparseTensorProto, TensorProto
from onnx.reference.op_run import to_array_extended as onnx_to_array_extended
from .c_reference_evaluator import CReferenceEvaluator, from_array_extended


def to_array_extended(
    tensor: Union[SparseTensorProto, TensorProto],
) -> Union[np.ndarray, "scipy.sparse.coo_matrix"]:  # noqa: F821
    """
    Overwrites function `onnx.reference.op_run.to_array_extended`
    to support sparse tensors.
    """
    if isinstance(tensor, TensorProto):
        return onnx_to_array_extended(tensor)
    if isinstance(tensor, SparseTensorProto):
        import scipy.sparse as sp

        shape = tuple(d for d in tensor.dims)
        indices = onnx_to_array_extended(tensor.indices)
        values = onnx_to_array_extended(tensor.values)
        if len(indices.shape) == 1:
            t = sp.csr_matrix(
                (values, indices, np.array([0, len(indices)], dtype=np.int64)),
                shape=(1, np.prod(shape)),
            )
            return t.reshape(shape)
        if len(indices.shape) == 2:
            t = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)
            return t
        raise RuntimeError(f"Unexpected indices shape: {indices.shape}.")
    raise TypeError(f"Unexpected type {type(tensor)}.")
