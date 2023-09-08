from onnx import FunctionProto, TensorProto
from onnx.helper import (
    make_function,
    make_node,
    make_opsetid,
    make_tensor,
)


def make_reshape_transpose_function_proto(
    domain: str, opset: int, index: int
) -> FunctionProto:
    """
    Creates the FunctionProto for function `ReshapeTranspose[index]`
    to reshape in two dimensions and transpose an input
    for operator Matmul.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param index: which input it is, 0 for left, 1 for right
    :return: FunctionProto

    The function takes 1 input and returns 1 output.
    """

    if index == 0:
        nodes = [
            make_node("Shape", ["x"], ["shape_x"]),
            make_node(
                "Constant",
                [],
                ["m1"],
                value=make_tensor("new_shape", TensorProto.INT64, [1], [-1]),
            ),
            make_node("Gather", ["shape_x", "m1"], ["last_dim"]),
            make_node("Concat", ["m1", "last_dim"], ["new_shape"], axis=0),
            make_node(
                "Reshape",
                ["x", "new_shape"],
                ["reshaped_name"],
            ),
            make_node(
                "Transpose",
                ["reshaped_name"],
                ["y"],
                perm=[1, 0],
            ),
        ]
    elif index == 1:
        raise NotImplementedError()
    else:
        raise ValueError(f"index must be 0 or 1 not {index}.")

    return make_function(
        domain,
        f"ReshapeTranspose{index}",
        ["x"],
        ["y"],
        nodes,
        opset_imports=[make_opsetid("", opset)],
    )


def make_reshape_transpose_back_function_proto(
    domain: str, opset: int, index: int
) -> FunctionProto:
    """
    Creates the FunctionProto for function `ReshapeTransposeBack[index]`
    to reshape with more two dimensions an input which was modified
    by `ReshapeTransposeBack[index]`

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param index: which input it is, 0 for left, 1 for right
    :return: FunctionProto

    The function takes 2 inputs `(x, shape)` and returns 1 output.
    """

    if index == 0:
        nodes = [
            make_node(
                "Constant",
                [],
                ["zero"],
                value=make_tensor("zero", TensorProto.INT64, [1], [0]),
            ),
            make_node(
                "Constant",
                [],
                ["m1"],
                value=make_tensor("m1", TensorProto.INT64, [1], [-1]),
            ),
            make_node(
                "Constant",
                [],
                ["m2"],
                value=make_tensor("m2", TensorProto.INT64, [1], [-2]),
            ),
            make_node("Slice", ["shape", "zero", "m2", "zero"], ["sliced"]),
            make_node("Gather", ["shape", "m2"], ["shm2"]),
            make_node("Concat", ["sliced", "shm2", "m1"], ["new_shape"], axis=0),
            make_node(
                "Reshape",
                ["x", "new_shape"],
                "y",
            ),
        ]
    elif index == 1:
        raise NotImplementedError()
    else:
        raise ValueError(f"index must be 0 or 1 not {index}.")

    return make_function(
        domain,
        f"ReshapeTransposeBack{index}",
        ["x", "shape"],
        ["y"],
        nodes,
        opset_imports=[make_opsetid("", opset)],
    )
