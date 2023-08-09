from logging import getLogger
from typing import Dict, List, Optional, Tuple
import numpy as np
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_19 as QuantizeLinear
from onnx.reference.op_run import to_array_extended
from ...reference.c_reference_evaluator import from_array_extended
from .onnx_graph_struct import Graph, Node


logger = getLogger("onnx-extended/transformer")


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


def _quantize_float8_matmul(
    node: Node,
    elem_type: int = TensorProto.FLOAT8E4M3FN,
    output_type: int = TensorProto.FLOAT,
) -> Optional[Tuple[List[Node], List[Node], Optional[Dict[str, int]]]]:
    """
    Quantize matrix multiplications.

    :param node: matrix multiplication
    :param elem_type: float 8 type to quantize into
    :param output_type: output type, result of the quantization
    :return: nodes to remove, nodes to add, new opsets
    """
    if node.op_type == "MatMul":
        removed = []
        added = []
        input_names = []
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
                    temp_name = node.parent.generate_name(f"{name}_tr")
                    added.append(
                        make_node("Transpose", [name], [temp_name], perm=[1, 0])
                    )
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
                )
                dql = Node(None, node.parent, proto)
                added.extend([dql.proto])
                input_names.append(dql.outputs)

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
                make_node("Mul", [input_names[0][1], input_names[1][1]], [scale_out])
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
        added.append(
            make_node(
                "GemmFloat8",
                gemm_inputs,
                node.outputs,
                domain="com.microsoft",
                rowMajor=1,
                dtype=output_type,
                transA=1,
            )
        )
        removed.append(node)
        return removed, added, {"com.microsoft": 1}

    raise NotImplementedError(
        f"Quantization into float 8 not yet implemented for {node.op_type!r}."
    )


def quantize_float8(
    graph: Graph,
    elem_type: int = TensorProto.FLOAT8E4M3FN,
    output_type: int = TensorProto.FLOAT,
    early_stop: int = -1,
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
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    The graph is modified inplace. Enables the logs gives a better idea of the progress.
    """
    main_opset = graph.get_opset("")
    if main_opset < 20:
        logger.info(
            "[quantize_float8] upgrade model from opset %d to %s", main_opset, 20
        )
        graph.upgrade_opsets({"": 20})
    new_opsets = {}
    to_add = []
    n_nodes = len(graph)
    n_changes = 0
    for index, node in enumerate(graph):
        if node.op_type in {"MatMul", "Gemm"}:
            logger.info("[quantize_float8] %d/%d quantize %s", index, n_nodes, node)
            res = _quantize_float8_matmul(node, elem_type, output_type)
            if res is None:
                continue
            rem, add = res[:2]
            to_add.append((rem, add))
            if len(res) >= 3 and res[2] is not None:
                n_changes += 1
                new_opsets.update(res[2])
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
    return graph
