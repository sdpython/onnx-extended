from logging import getLogger
from typing import Dict, List, Optional, Tuple
import numpy as np
from onnx import AttributeProto, FunctionProto, NodeProto, TensorProto
from onnx.helper import (
    make_function,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.custom_element_types import float8e4m3fn
from onnx.reference.op_run import to_array_extended
from onnx.onnx_cpp2py_export.defs import SchemaError

try:
    from onnx.reference.ops.op_cast import Cast_19 as Cast
    from onnx.reference.ops.op_quantize_linear import (
        QuantizeLinear_19 as QuantizeLinear,
    )
except ImportError:
    from onnx.reference.ops.op_cast import Cast
    from onnx.reference.ops.op_quantize_linear import QuantizeLinear
from ...reference.c_reference_evaluator import from_array_extended
from ...validation.cython.fp8 import cast_float32_to_e4m3fn
from .errors import QuantizationError
from .onnx_graph_struct import _get_shape, Graph, Node, NodeKind


logger = getLogger("onnx-extended/transformer")


class TransformResults:
    """
    Output of a function transforming a graph.

    :param removed_nodes: node to remove from the graph
    :param added_nodes: node to add to the graph
    :param new_opsets: opsets to update
    :param local_functions: necessary functions to add to the graph
    """

    def __init__(
        self,
        removed_nodes: List[Node],
        added_nodes: List[NodeProto],
        new_opsets: Optional[Dict[str, int]] = None,
        local_functions: Optional[List[FunctionProto]] = None,
    ):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.new_opsets = new_opsets or {}
        self.local_functions = local_functions or []


def estimation_quantization_scale(
    coef: np.array,
    to: int = TensorProto.FLOAT8E4M3FN,
    threshold: float = 0.99999,
) -> Tuple[float, float]:
    """
    Estimates the scale parameter for the quantization to float 8 assuming
    the distribution of the coefficients is gaussian.
    """

    if to in (
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    ):
        if to == TensorProto.FLOAT8E4M3FN:
            fct = float8e4m3_to_float32
        elif to == TensorProto.FLOAT8E4M3FNUZ:

            def fct(x):
                return float8e4m3_to_float32(x, uz=True)

        elif to == TensorProto.FLOAT8E5M2:
            fct = float8e5m2_to_float32
        elif to == TensorProto.FLOAT8E5M2FNUZ:

            def fct(x):
                return float8e5m2_to_float32(x, uz=True, fn=True)

        else:
            raise ValueError(f"Unexpected to={to!r}.")

        float8 = [fct(i) for i in range(0, 256)]
        quant_float = [f for f in float8 if not np.isnan(f) and not np.isinf(f)]
        std_coef = np.mean(coef.ravel() ** 2) ** 0.5
        std_quant = np.std(np.array(quant_float, dtype=np.float32))
        zero = 0.0
        scale = std_quant.astype(coef.dtype) / std_coef.astype(coef.dtype)
    elif to == TensorProto.UINT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 255 / (qu[1] - qu[0])
        zero = qu[0] * scale
    elif to == TensorProto.INT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 254 / (qu[1] - qu[0])
        zero = (qu[0] + qu[1]) / 2 * scale
    else:
        raise ValueError(f"Unexpected quantization type for to={to}.")

    return np.array(1.0 / scale, dtype=coef.dtype), -zero


def quantize_weights(
    node: Node, elem_type: int, transpose: bool = False
) -> Tuple[Node, Node, Node]:
    """
    Quantizes a tensor into a tensor of element type *elem_type*.

    :param node: Node to quantize
    :param elem_type: element type
    :param transpose: transpose the weight before doing it
    :return: three new nodes, quantized weights, scale, zero point
    """
    tensor = node.get_tensor()
    values = to_array_extended(tensor)
    if transpose:
        values = values.T
    scale, zp = estimation_quantization_scale(values, to=elem_type)
    zpt = make_tensor("zp", elem_type, [], [zp])
    zpa = to_array_extended(zpt)

    if elem_type == TensorProto.FLOAT8E4M3FN:
        new_values = cast_float32_to_e4m3fn(values / scale).astype(float8e4m3fn)
    else:
        new_values = QuantizeLinear.eval(values, scale, zpa)

    new_tensor = from_array_extended(
        new_values, name=node.parent.generate_name(node.outname)
    )
    node_weight = node.create_with_new_values(new_tensor)

    new_scale = from_array_extended(
        scale, name=node.parent.generate_name(f"{node.outname}_scale")
    )
    node_scale = node.create_with_new_values(new_scale)

    new_zp = from_array_extended(
        zpa, name=node.parent.generate_name(f"{node.outname}_zp")
    )
    node_zp = node.create_with_new_values(new_zp)

    return node_weight, node_scale, node_zp


def make_dynamic_quantize_linear_function(
    domain: str, opset: int, to: Optional[int] = None
) -> FunctionProto:
    """
    Creates the FunctionProto for a function doing a quantization to float 8.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param to: if None, the function has an attribute,
        otherwise, it is replaced by the given value
    :return: FunctionProto

    The function takes 1 input and returns 3 outputs.
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
        cast,
        make_node("Mul", ["x", "x"], ["xsquare"]),
        make_node("ReduceMean", ["xsquare"], ["Dev"], keepdims=0),
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


def _quantize_float8_matmul(
    node: Node,
    elem_type: int,
    output_type: int,
    version: str,
    opset: int,
    local_functions: Optional[Dict[str, FunctionProto]] = None,
) -> Optional[TransformResults]:
    """
    Quantize matrix multiplications.

    :param node: matrix multiplication
    :param elem_type: float 8 type to quantize into
    :param output_type: output type, result of the quantization
    :param version: `'onnxruntime'` to use operators from onnx and onnxruntime,
        `'onnx-extended'` to use experimental operators
    :param opset: main opset (used to specify local functions)
    :param local_functions: None to avoid using local functions,
        otherwise a dictionary with the existing local functions to
        add to the model
    :param quiet: True to silently skip failing nodes
    :return: nodes to remove, nodes to add, new opsets
    """
    if version == "onnxruntime":
        domain_dq = ""
        domain_gemm = "com.microsoft"
        op_gemm = "GemmFloat8"
    elif version == "onnx-extended":
        domain_dq = "onnx_extented.ortops.tutorial.cpu"
        domain_gemm = "onnx_extented.ortops.tutorial.cuda"
        op_gemm = "CustomGemmFloat8E4M3FN"
    else:
        raise ValueError(f"Unexpected value {version!r} for version.")
    if local_functions is not None:
        domain_dq = "local.quant.domain"
    if node.op_type == "MatMul":
        removed = []
        added = []
        input_names = []
        was_reshaped = [None, None]
        m1 = None
        for index, name in enumerate(node.inputs):
            if node.parent.is_constant(name):
                # Quantized constant weights
                cst = node.parent.get_node_producer(name)
                weight, scale, zero_point = quantize_weights(
                    cst, elem_type, transpose=index == 0
                )
                added.extend([weight.proto, scale.proto, zero_point.proto])
                input_names.append([weight.outname, scale.outname, zero_point.outname])
                removed.append(cst)
            else:
                # Add DynamicQuantizeLinear
                if index == 0:
                    # transposition is needed for the first input
                    shape = node.parent.get_shape(name)
                    if shape is None:
                        raise QuantizationError(
                            f"Shape is unknown for result {name!r} in node {node}. "
                            f"This input cannot be transposed with certainty."
                        )

                    # Let's reshape if the dimension is != 2
                    if len(shape) == 2:
                        # no need
                        reshaped_name = name
                    else:
                        # The input needs to be reshaped.
                        reshaped_name = node.parent.generate_name(f"{name}_sh")
                        shape_name = node.parent.generate_name(f"{name}_sh")
                        new_shape = node.parent.generate_name(f"{name}_nsh")
                        last_dim = node.parent.generate_name(f"{name}_ldim")
                        m1 = node.parent.generate_name(f"{name}_m1")
                        was_reshaped[index] = shape_name
                        added.extend(
                            [
                                make_node("Shape", [name], [shape_name]),
                                make_node(
                                    "Constant",
                                    [],
                                    [m1],
                                    value=make_tensor(
                                        new_shape, TensorProto.INT64, [1], [-1]
                                    ),
                                ),
                                make_node("Gather", [shape_name, m1], [last_dim]),
                                make_node(
                                    "Concat", [m1, last_dim], [new_shape], axis=0
                                ),
                                make_node(
                                    "Reshape",
                                    [name, new_shape],
                                    [reshaped_name],
                                    name=node.parent.generate_node_name(
                                        f"resh8_{name}"
                                    ),
                                ),
                            ]
                        )

                    tr_name = node.parent.generate_name(f"{name}_tr")
                    added.append(
                        make_node(
                            "Transpose",
                            [reshaped_name],
                            [tr_name],
                            perm=[1, 0],
                            name=node.parent.generate_node_name(f"tra8_{name}"),
                        )
                    )
                    temp_name = tr_name
                else:
                    # no transposition for the other input
                    temp_name = name
                new_name = node.parent.generate_name(f"{name}_f8")
                scale = node.parent.generate_name(f"{name}_scale")
                zero_point = node.parent.generate_name(f"{name}_zp")
                proto = make_node(
                    "DynamicQuantizeLinear",
                    [temp_name],
                    [new_name, scale, zero_point],
                    to=elem_type,
                    domain=domain_dq,
                    name=node.parent.generate_node_name(f"dql8_{name}"),
                )
                dql = Node(None, node.parent, proto, NodeKind.NODE)
                added.extend([dql.proto])
                input_names.append(dql.outputs)
                if (
                    domain_dq == "local.quant.domain"
                    and (domain_dq, "DynamicQuantizeLinear") not in local_functions
                ):
                    # use local functions
                    local_functions[
                        domain_dq, "DynamicQuantizeLinear"
                    ] = make_dynamic_quantize_linear_function(
                        domain=domain_dq, opset=opset
                    )

        if was_reshaped[0] is not None and was_reshaped[1] is not None:
            raise QuantizationError(
                f"MatMul cannot be replaced by operator Gemm as both inputs "
                f"are not matrices of shapes {was_reshaped}."
            )
        if output_type in {
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }:
            # output is quantized, there is a need for a scale
            scale_out = node.parent.generate_name(f"{name}_scaleout")
            added.append(
                make_node(
                    "Mul",
                    [input_names[0][1], input_names[1][1]],
                    [scale_out],
                    name=node.parent.generate_node_name(f"mul8_{name}"),
                )
            )
        else:
            # output is not quantized, no need for an output scale
            scale_out = ""
        gemm_inputs = [
            input_names[0][0],  # A
            input_names[1][0],  # B
            "",  # C
            input_names[0][1],  # scaleA
            input_names[1][1],  # scaleB
            scale_out,  # scaleR
        ]
        if was_reshaped[0] is not None or was_reshaped[1] is not None:
            gemm_outputs = [node.parent.generate_name(f"{name}_gemm")]
            do_reshape = True
        else:
            gemm_outputs = node.outputs
            do_reshape = False
        added.append(
            make_node(
                op_gemm,
                gemm_inputs,
                gemm_outputs,
                rowMajor=0,
                dtype=output_type,
                transA=1,
                domain=domain_gemm,
                computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            )
        )
        if do_reshape:
            # One of the inputs had 3 dimensions.
            if was_reshaped[0] is not None:
                assert m1 is not None
                sliced = node.parent.generate_name(f"{name}_sli")
                new_shape = node.parent.generate_name(f"{name}_sh1")
                zero = node.parent.generate_name("zero")
                m2 = node.parent.generate_name(f"{name}_shm2")
                shm2 = node.parent.generate_name("m2")
                sh1 = was_reshaped[0]
                added.extend(
                    [
                        make_node(
                            "Constant",
                            [],
                            [zero],
                            value=make_tensor(new_shape, TensorProto.INT64, [1], [0]),
                        ),
                        make_node(
                            "Constant",
                            [],
                            [m2],
                            value=make_tensor(new_shape, TensorProto.INT64, [1], [-2]),
                        ),
                        make_node("Slice", [sh1, zero, m2, zero], [sliced]),
                        make_node("Gather", [sh1, m2], [shm2]),
                        make_node("Concat", [sliced, shm2, m1], [new_shape], axis=0),
                        make_node(
                            "Reshape",
                            [gemm_outputs[0], new_shape],
                            node.outputs,
                            name=node.parent.generate_node_name(f"resh82_{name}"),
                        ),
                    ]
                )
            else:
                raise NotImplementedError(
                    f"MatMul cannot be replaced by operator Gemm, "
                    f"shapes are {was_reshaped}."
                )
        removed.append(node)
        opsets = {domain_gemm: 1}
        if domain_dq != "":
            opsets.update({domain_dq: 1})
        return TransformResults(
            removed_nodes=removed, added_nodes=added, new_opsets=opsets
        )

    raise NotImplementedError(
        f"Quantization into float 8 not yet implemented for {node.op_type!r}."
    )


def quantize_float8(
    graph: Graph,
    elem_type: int = TensorProto.FLOAT8E4M3FN,
    output_type: int = TensorProto.FLOAT,
    early_stop: int = -1,
    version: str = "onnxruntime",
    local_function: bool = False,
    quiet: bool = False,
) -> Optional[Graph]:
    """
    Transforms a graph to introduce quantized weights.
    This transformation requires opset 20. The graph is
    upgraded if the main opset is below. It is better to do
    it before calling this function.

    :param graph: Graph
    :param elem_type: quantization type
    :param output_type: output type
    :param early_stop: -1 to go through all nodes or a value `n > 0`
        to stop after n changes
    :param version: `'onnxruntime'` to use operators from onnx and onnxruntime,
        `'onnx-extended'` to use experimental operators
    :param local_function: use local function to inline DynamicQuantizeLinear
    :param quiet: catch exception and silently skip failing nodes
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    The graph is modified inplace.
    Enables the logs gives a better idea of the progress.
    """
    main_opset = graph.get_opset("")
    if not local_function and main_opset < 20 and version == "onnxruntime":
        logger.info(
            "[quantize_float8] upgrade model from opset %d to %s", main_opset, 20
        )
        graph.upgrade_opsets({"": 20})
        main_opset = 20
    elif local_function and main_opset < 19:
        logger.info(
            "[quantize_float8] upgrade model from opset %d to %s", main_opset, 19
        )
        graph.upgrade_opsets({"": 19})
        main_opset = 19

    if len(graph.functions) > 0:
        raise NotImplementedError("Quantization of local functions is not implemented.")
    local_functions = graph.functions.copy() if local_function else None
    n_local_functions = 0 if local_functions is None else len(local_functions)
    new_opsets = {}
    to_add = []
    n_nodes = len(graph)
    n_changes = 0
    for index, node in enumerate(graph):
        if node.op_type in {"MatMul", "Gemm"}:
            logger.info("[quantize_float8] %d/%d quantize %s", index, n_nodes, node)
            try:
                results = _quantize_float8_matmul(
                    node,
                    elem_type,
                    output_type,
                    version=version,
                    opset=main_opset,
                    local_functions=local_functions,
                )
            except (QuantizationError, NotImplementedError) as e:
                if quiet:
                    logger.warn(
                        "[quantize_float8] %d/%d failed to quantize due to %s",
                        index,
                        n_nodes,
                        e,
                    )
                    continue
                raise e

            if results is None:
                continue
            rem, add = results.removed_nodes, results.added_nodes
            to_add.append((rem, add))
            if len(results.new_opsets) > 0:
                n_changes += 1
                new_opsets.update(results.new_opsets)
            if early_stop > 0 and n_changes >= early_stop:
                break

    if len(to_add) == 0:
        return None

    for rem, add in to_add:
        for r in rem:
            logger.debug("[quantize_float8] del %s", r)
        added = graph.replace_nodes([r.index for r in rem], add, new_opsets)
        for a in added:
            logger.debug("[quantize_float8] add %s", a)

    graph.simplify()
    if local_functions is not None and len(local_functions) > n_local_functions:
        graph.add_functions(local_functions.values())
    return graph


def _cast_constant(
    node: Node,
    from_type: int,
    to_type: int,
) -> Optional[TransformResults]:
    """
    Converts a node if it is a tensor of element type *from_type*
    and cast it into *to_type*.

    :param node: node
    :param from_type: element type to replace
    :param to_type: type to cast into
    :return: TransformResults
    """
    op_type = node.op_type

    if op_type in {"Constant", "initializer"}:
        cst = node.get_tensor()
        if cst.data_type != from_type:
            return None

        arr = to_array_extended(cst)
        try:
            cast = Cast.eval(arr, to=to_type)
        except SchemaError:
            # cast does not work
            np_type = tensor_dtype_to_np_dtype(to_type)
            cast = arr.astype(np_type)
        if op_type == "initializer":
            return TransformResults(
                removed_nodes=[node],
                added_nodes=[from_array_extended(cast, name=node.proto.name)],
            )
        if op_type == "Constant":
            return TransformResults(
                removed_nodes=[node],
                added_nodes=[
                    make_node(
                        "Constant",
                        [],
                        node.outputs,
                        value=from_array_extended(cast, name=node.outputs[0]),
                    )
                ],
            )
    if op_type in {"input", "output"}:
        if isinstance(node.proto, str):
            # A FunctionProto input, nothing to do.
            return None
        ttype = node.proto.type
        if not ttype.tensor_type or ttype.tensor_type.elem_type != from_type:
            return None
        return TransformResults(
            removed_nodes=[node],
            added_nodes=[
                make_tensor_value_info(
                    node.proto.name,
                    to_type,
                    shape=_get_shape(ttype),
                    doc_string=node.proto.doc_string,
                    shape_denotation=ttype.denotation,
                )
            ],
        )
    if op_type in {"Cast"}:
        to = node.getattr("to", int)
        if to != from_type:
            return None
        return TransformResults(
            removed_nodes=[node],
            added_nodes=[make_node("Cast", node.inputs, node.outputs, to=to_type)],
        )

    raise RuntimeError(f"Unexpected node type {op_type!r}.")


def cast_constant(
    graph: Graph,
    from_type: int = TensorProto.FLOAT,
    to_type: int = TensorProto.FLOAT16,
    quiet: bool = False,
) -> Optional[Graph]:
    """
    Converts all constants and initializers to the same type.
    It also modifies the input.

    :param graph: Graph
    :param from_type: type of the constants to convert
    :param to_type: new type for the constants
    :param quiet: catch exception and silently skip failing nodes
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    The graph is modified inplace.
    Enables the logs gives a better idea of the progress.
    """
    if len(graph.functions) > 0:
        raise NotImplementedError("Conversion of local functions is not implemented.")

    to_add = []
    n_nodes = len(graph)
    for index, node in enumerate(graph):
        if node.op_type in {"Constant", "initializer", "input", "output", "Cast"}:
            logger.info("[cast_constant] %d/%d convert %s", index, n_nodes, node)
            results = _cast_constant(node, from_type, to_type)
            if results is None:
                continue
            rem, add = results.removed_nodes, results.added_nodes
            to_add.append((rem, add))

    if len(to_add) == 0:
        return None

    for rem, add in to_add:
        for r in rem:
            logger.debug("[cast_constant] del %s", r)
        added = graph.replace_nodes([r.index for r in rem], add)
        for a in added:
            logger.debug("[cast_constant] add %s", a)

    return graph.simplify()
