# source: https://github.com/onnx/onnx/blob/main/onnx/helper.py
from typing import Sequence
from .cpu._onnx2py import OperatorSetIdProto, TypeProto


def make_operatorsetid(
    domain: str,
    version: int,
) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Args:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid


def make_tensor_type_proto(
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    shape_denotation: list[str] | None = None,
) -> TypeProto:
    """Makes a Tensor TypeProto based on the data type and shape."""
    type_proto = TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = elem_type
    tensor_shape_proto = tensor_type_proto.shape

    if shape is not None:
        tensor_shape_proto.dim.extend([])

        if shape_denotation and len(shape_denotation) != len(shape):
            raise ValueError(
                "Invalid shape_denotation. Must be of the same length as shape."
            )

        for i, d in enumerate(shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    f"Invalid item in shape: {d}. Needs to be of int or str."
                )

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto
