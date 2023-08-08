from logging import getLogger
from typing import List, Optional, Tuple
from onnx import TensorProto
from onnx.helper import make_node
from .onnx_graph_struct import Graph, Node, NodeSet

logger = getLogger("onnx-extended/transformer")


def quantize_weights(node: Node, elem_type: int):
    """
    Quantizes a tensor into a tensor of element type *elem_type*.

    :param node: Node to quantize
    :param elem_type: element type
    :return: three new nodes, quantized weights, scale, zero point
    """
    raise NotImplementedError("Not implemented yet.")


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
            make_node("GemmFloat8", [added], node.outputs, domain="com.microsoft"),
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
        graph.update_opset({"": 20})
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
