from typing import Dict, Optional
from onnx import AttributeProto, FunctionProto, TensorProto
from onnx.helper import (
    make_function,
    make_node,
    make_opsetid,
    make_tensor,
)


def make_dynamic_quantize_linear_function_proto(
    domain: str, opset: int, to: Optional[int] = None
) -> FunctionProto:
    """
    Creates the FunctionProto for function `DynamicQuantizeLinear`
    doing a quantization to float 8.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param to: if None, the function has an attribute,
        otherwise, it is replaced by the given value
    :return: FunctionProto

    The function takes 1 input and returns 3 outputs like
    operator `DynamicQuantizeLinear
    <https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html>`_.
    It has one attribute *to* which specified the quantized type.
    """
    normalization_values = list(
        {
            TensorProto.FLOAT8E4M3FN: 100.057724,
            TensorProto.FLOAT8E4M3FNUZ: 54.26635,
            TensorProto.FLOAT8E5M2: 9535.286,
            TensorProto.FLOAT8E5M2FNUZ: 9403.499,
        }.items()
    )

    if to is None:
        cast = make_node("Cast", ["zerof"], ["Zeropoint"])
        att = AttributeProto()
        att.name = "to"
        att.ref_attr_name = "to"
        att.type = AttributeProto.INT
        cast.attribute.append(att)

        cst = make_node("Constant", [], ["vto"])
        att = AttributeProto()
        att.name = "value_int"
        att.ref_attr_name = "to"
        att.type = AttributeProto.INT
        cst.attribute.append(att)
    else:
        cast = make_node("Cast", ["zerof"], ["Zeropoint"], to=to)
        cst = make_node("Constant", [], ["vto"], value_int=to)

    nodes = [
        make_node(
            "Constant",
            [],
            ["zerof"],
            value=make_tensor("zerof", TensorProto.FLOAT, [], [0]),
        ),
        make_node(
            "Constant",
            [],
            ["newshape"],
            value=make_tensor("newshape", TensorProto.INT64, [1], [-1]),
        ),
        make_node("CastLike", ["zerof", "x"], ["zero"]),
        cast,
        make_node("IsNaN", ["x"], ["nanxp"]),
        make_node("Not", ["nanxp"], ["nanx"]),
        make_node("CastLike", ["nanx", "x"], ["nanxc"]),
        make_node("Where", ["nanx", "x", "zero"], ["xf"]),
        make_node("Mul", ["xf", "xf"], ["xsquare"]),
        make_node("ReduceSum", ["xsquare"], ["Num"], keepdims=0),
        make_node("ReduceSum", ["nanxc"], ["Den"], keepdims=0),
        make_node("Div", ["Num", "Den"], ["Dev"]),
        make_node("Sqrt", ["Dev"], ["Scale"]),
        cst,
        make_node("Reshape", ["vto", "newshape"], ["vtotensor"]),
        make_node(
            "LabelEncoder",
            ["vtotensor"],
            ["stdftensor"],
            keys_int64s=[v[0] for v in normalization_values],
            values_floats=[v[1] for v in normalization_values],
            domain="ai.onnx.ml",
        ),
        make_node("ReduceSum", ["stdftensor"], ["stdf"], keepdims=0),
        make_node("CastLike", ["stdf", "Scale"], ["std"]),
        make_node("Div", ["Scale", "std"], ["ScaleScaled"]),
        make_node("QuantizeLinear", ["x", "ScaleScaled", "Zeropoint"], ["y"]),
    ]
    return make_function(
        domain,
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "ScaleScaled", "Zeropoint"],
        nodes,
        opset_imports=[make_opsetid("", opset), make_opsetid("ai.onnx.ml", 2)],
        attributes=["to"],
    )


def make_simple_dynamic_quantize_linear_function_proto(
    domain: str, opset: int, to: int = TensorProto.FLOAT8E4M3FN
) -> FunctionProto:
    """
    Creates the FunctionProto for function `SimpleDynamicQuantizeLinear`
    doing a quantization to float 8. A suffix is added to the function name
    to tell which type is used for the quantization. It does not
    support nan values.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param to: type to quantize into, it is hardcoded
    :return: FunctionProto

    The function takes 1 input and returns 3 outputs like
    operator `DynamicQuantizeLinear
    <https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html>`_.
    It has one attribute *to* which specified the quantized type.
    """
    normalization_values: Dict[int, float] = {
        TensorProto.FLOAT8E4M3FN: 100.057724,
        TensorProto.FLOAT8E4M3FNUZ: 54.26635,
        TensorProto.FLOAT8E5M2: 9535.286,
        TensorProto.FLOAT8E5M2FNUZ: 9403.499,
    }
    suffix: Dict[int, str] = {
        TensorProto.FLOAT8E4M3FN: "E4M3FN",
        TensorProto.FLOAT8E4M3FNUZ: "E4M3FNUZ",
        TensorProto.FLOAT8E5M2: "E5M2",
        TensorProto.FLOAT8E5M2FNUZ: "E5M2FNUZ",
    }

    nodes = [
        make_node(
            "Constant",
            [],
            ["zerof"],
            value=make_tensor("zerof", TensorProto.FLOAT, [], [0]),
        ),
        make_node("Cast", ["zerof"], ["Zeropoint"], to=to),
        make_node(
            "Constant",
            [],
            ["stdf"],
            value=make_tensor(
                "stdf", TensorProto.FLOAT, [], [normalization_values[to]]
            ),
        ),
        make_node(
            "Constant",
            [],
            ["newshape"],
            value=make_tensor("newshape", TensorProto.INT64, [1], [-1]),
        ),
        make_node("Mul", ["x", "x"], ["xsquare"]),
        make_node("ReduceMean", ["xsquare"], ["Dev"], keepdims=0),
        make_node("Sqrt", ["Dev"], ["Scale"]),
        make_node("CastLike", ["stdf", "Scale"], ["std"]),
        make_node("Div", ["Scale", "std"], ["ScaleScaled"]),
        make_node("QuantizeLinear", ["x", "ScaleScaled", "Zeropoint"], ["y"]),
    ]
    return make_function(
        domain,
        f"DynamicQuantizeLinear{suffix[to]}",
        ["x"],
        ["y", "ScaleScaled", "Zeropoint"],
        nodes,
        opset_imports=[make_opsetid("", opset)],
        attributes=["to"],
    )
