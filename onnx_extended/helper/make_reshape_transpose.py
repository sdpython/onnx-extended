from onnx import FunctionProto, TensorProto
from onnx.helper import (
    make_function,
    make_node,
    make_opsetid,
    make_tensor,
)


def make_matmul_reshape_transpose_function_proto(
    domain: str, opset: int, index: int, transpose: bool
) -> FunctionProto:
    """
    Creates the FunctionProto for function
    `MatMulReshapeTranspose[index]`
    to reshape in two dimensions and transpose an input
    for operator Matmul. The transposition is optional.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param index: which input it is, 0 for left, 1 for right
    :param transpose: transpose the input
    :return: FunctionProto

    The function takes 1 input and returns 1 output.
    See :func:`quantize_float8_matmul`.
    """

    if index == 0:
        nodes = [
            make_node("Shape", ["x"], ["shape_x"]),
            make_node(
                "Constant",
                [],
                ["m1"],
                value=make_tensor("m1", TensorProto.INT64, [1], [-1]),
            ),
            make_node("Gather", ["shape_x", "m1"], ["last_dim"]),
            make_node("Concat", ["m1", "last_dim"], ["new_shape"], axis=0),
            make_node("Reshape", ["x", "new_shape"], ["reshaped_name"]),
        ]
    elif index == 1:
        # values = (
        #     values.reshape((-1,) + values.shape[-2:])
        #     .transpose((1, 0, 2))
        #     .reshape((values.shape[-2], -1))
        # )
        nodes = [
            make_node("Shape", ["x"], ["shape_x"]),
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
            make_node(
                "Constant",
                [],
                ["m21"],
                value=make_tensor("m2", TensorProto.INT64, [2], [-2, -1]),
            ),
            make_node("Gather", ["shape_x", "m21"], ["last_dims"]),
            make_node("Concat", ["m1", "last_dims"], ["new_shape"], axis=0),
            make_node("Reshape", ["x", "new_shape"], ["xsh"]),
            make_node("Transpose", ["xsh"], ["xtr"], perm=[1, 0, 2]),
            make_node("Gather", ["shape_x", "m2"], ["bldims"]),
            make_node("Concat", ["bldims", "m1"], ["new_shape2"], axis=0),
            make_node("Reshape", ["xtr", "new_shape2"], ["reshaped_name"]),
        ]
    else:
        raise ValueError(f"index must be 0 or 1 not {index}.")

    if transpose:
        nodes.append(make_node("Transpose", ["reshaped_name"], ["y"], perm=[1, 0]))
    else:
        nodes.append(make_node("Identity", ["reshaped_name"], ["y"]))

    return make_function(
        domain,
        f"MatMulReshapeTranspose{'T' if transpose else 'N'}{index}",
        ["x"],
        ["y"],
        nodes,
        opset_imports=[make_opsetid("", opset)],
    )


def make_matmul_reshape_transpose_back_function_proto(
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
    See :func:`quantize_float8_matmul`.
    """

    if index == 0:
        # res.reshape(a.shape[:-2] + (-1, res.shape[-1]))
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
            make_node("Reshape", ["x", "new_shape"], ["y"]),
        ]
    elif index == 1:
        # final = (
        #     res.reshape(a.shape[0], -1, b.shape[-1])
        #     .transpose((1, 0, 2))
        #     .reshape(b.shape[:-2] + (-1, b.shape[-1]))
        # )
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
            make_node("Shape", ["x"], ["shape_x"]),
            make_node("Gather", ["shape_x", "zero"], ["d0"]),
            make_node("Gather", ["shape", "m1"], ["last_dim"]),
            make_node("Concat", ["d0", "m1", "last_dim"], ["new_shape"], axis=0),
            make_node("Reshape", ["x", "new_shape"], ["xsh"]),
            make_node("Transpose", ["xsh"], ["xtr"], perm=[1, 0, 2]),
            make_node("Slice", ["shape", "zero", "m2", "zero"], ["sliced"]),
            make_node("Concat", ["sliced", "m1", "last_dim"], ["final_shape"], axis=0),
            make_node("Reshape", ["xtr", "final_shape"], ["y"]),
        ]

    else:
        raise ValueError(f"index must be 0 or 1 not {index}.")

    return make_function(
        domain,
        f"MatMulReshapeTransposeBack{index}",
        ["x", "shape"],
        ["y"],
        nodes,
        opset_imports=[make_opsetid("", opset)],
    )
