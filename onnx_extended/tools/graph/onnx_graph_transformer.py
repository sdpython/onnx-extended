from logging import getLogger
from typing import List, Optional, Tuple
import numpy as np
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_19 as QuantizeLinear
from onnx.reference.op_run import to_array_extended
from ...reference.c_reference_evaluator import from_array_extended
from .onnx_graph_struct import Graph, Node, NodeSet


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


def quantize_weights(node: Node, elem_type: int) -> Node:
    """
    Quantizes a tensor into a tensor of element type *elem_type*.

    :param node: Node to quantize
    :param elem_type: element type
    :return: three new nodes, quantized weights, scale, zero point
    """
    tensor = node.get_tensor()
    values = to_array_extended(tensor)
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
    node: Node, elem_type: int = TensorProto.FLOAT8E4M3FN
) -> Optional[Tuple[List[Node], List[Node]]]:
    """
    Quantize matrix multiplications.

    :param node: matrix multiplication
    :param elem_type: float 8 type to quantize into
    :return: nodes to remove, nodes to add
    """
    if node.op_type == "MatMul":
        removed = []
        added = []
        for name in node.inputs:
            if node.parent.is_constant(name):
                # Quantized constant weights
                cst = node.parent.get_node_producer(name)
                weight, scale, zero_point = quantize_weights(cst, elem_type)
                added.extend([weight, scale, zero_point])
                removed.append(cst)
            else:
                # Add DynamicQuantizeLinear
                new_name = node.parent.generate_name(f"{name}_f8")
                scale = node.parent.generate_name(f"{name}_scale")
                zero_point = node.parent.generate_name(f"{name}_zp")
                proto = make_node(
                    "DynamicQuantizeLinear",
                    [name],
                    [new_name, scale, zero_point],
                    to=elem_type,
                )
                dql = Node(None, node.parent, proto)
                added.extend([dql, scale, zero_point])

        new_node = Node(
            None,
            node.parent,
            make_node(
                "GemmFloat8",
                [a.outname for a in added],
                node.outputs,
                domain="com.microsoft",
            ),
        )
        added.append(new_node)
        removed.append(node)
        return removed, added

    raise NotImplementedError(
        f"Quantization into float 8 not yet implemented for {node.op_type!r}."
    )


def quantize_float8(
    graph: Graph, elem_type: int = TensorProto.FLOAT8E4M3FN
) -> Optional[Graph]:
    """
    Transforms a graph to introduce quantized weights.
    This transformation requires opset 20. The graph is
    upgraded if the main opset is below. It is better to do
    it before calling this function.

    :param graph: Graph
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    """
    main_opset = graph.get_opset("")
    if main_opset < 20:
        logger.info(
            "[quantize_float8] upgrade model from opset %d to %s", main_opset, 20
        )
        graph.upgrade_opsets({"": 20})
    to_add = []
    for node in graph:
        if node.op_type in {"MatMul", "Gemm"}:
            logger.info("[quantize_float8] quantize %s", node)
            res = _quantize_float8_matmul(node, elem_type)
            if res is None:
                continue
            rem, add = res
            to_add.append((rem, add))

    if len(to_add) == 0:
        return None

    for rem, add in to_add:
        graph.replace([r.index for r in rem], NodeSet(to_add))

    graph.simplify()
    return graph
