from typing import Any, Dict, Optional, Union
from onnx import AttributeProto, ModelProto, NodeProto, GraphProto, FunctionProto
from onnx.helper import make_model, make_node, make_graph, make_opsetid


def has_subgraph(node: NodeProto) -> bool:
    """
    Tells if a node has a subgraph as an attribute.
    """
    for att in node.attribute:
        if att.type == AttributeProto.GRAPH:
            return True
    return False


def get_node_attribute(node: NodeProto, name: str) -> AttributeProto:
    """
    Returns the value of one attribute.

    :param node: node
    :param name: attribute name
    :return: value
    """
    for att in node.attribute:
        if att.name == name:
            return att
    raise KeyError(
        f"Unable to find {name!r} among {list(att.name for att in node.attribute)}."
    )


def change_onnx_operator_domain(
    onx: Union[ModelProto, GraphProto, FunctionProto],
    op_type: str,
    op_domain: str = "",
    new_op_type: Optional[str] = None,
    new_op_domain: Optional[str] = None,
    new_opset: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> Union[ModelProto, GraphProto, FunctionProto]:
    """
    Replaces an operator by another one in the same domain
    or another one.

    :param onx: proto to modify
    :param op_type: operator to look for
    :param op_domain: domain to look for
    :param new_op_type: new operator name or None for the same name
    :param new_op_domain: new domain name or None the for the same domain
    :param new_opset: new opset for the new domain
    :param kwargs: modified parameters, set it to None to remove them
    :return: same type as the input

    The function is not recursive yet.
    """

    def change_node(node):
        atts = []
        new_kwargs = {}
        for att in node.attribute:
            if att.name in kwargs:
                v = kwargs[att.name]
                if v is None:
                    continue
                new_kwargs[att.name] = v
                continue
            atts.append(att)
        for k, v in kwargs.items():
            if v is None or k in new_kwargs:
                continue
            new_kwargs[k] = v
        new_node = make_node(
            new_op_type or node.op_type,
            node.input,
            node.output,
            domain=new_op_domain or node.domain,
            **new_kwargs,
        )
        if len(atts) > 0:
            new_node.attribute.extend(atts)
        return new_node

    if isinstance(onx, GraphProto):
        new_nodes = []
        modified = False
        for node in onx.node:
            if has_subgraph(node):
                raise NotImplementedError(
                    f"The function is not recursive yet and cannot "
                    f"handle node {node.op_type!r} from domain "
                    f"{node.domain!r}."
                )
            if node.op_type == op_type and node.domain == op_domain:
                new_node = change_node(node)
                new_nodes.append(new_node)
                modified = True
                continue
            new_nodes.append(node)
        if not modified:
            return onx
        return make_graph(
            new_nodes,
            onx.name,
            onx.input,
            onx.output,
            onx.initializer,
            onx.sparse_initializer,
        )

    if isinstance(onx, FunctionProto):
        raise NotImplementedError()

    if not isinstance(onx, ModelProto):
        raise TypeError(f"Unexpected type for onx {type(onx)}.")

    new_graph = change_onnx_operator_domain(
        onx.graph,
        op_type=op_type,
        op_domain=op_domain,
        new_opset=new_opset,
        new_op_type=new_op_type,
        new_op_domain=new_op_domain,
        **kwargs,
    )
    if id(new_graph) == id(onx.graph):
        # no change
        return onx

    if new_op_domain is None:
        new_op_domain = op_domain
    if new_op_domain == op_domain and new_opset is not None:
        raise ValueError(
            f"If new_op_domain==domain=={new_op_domain!r}, "
            f"new_opset must be None not {new_opset}."
        )
    opsets = list(onx.opset_import)
    if new_op_domain != op_domain:
        opsets.append(make_opsetid(new_op_domain, new_opset or 1))

    new_model = make_model(
        new_graph,
        functions=onx.functions,
        ir_version=onx.ir_version,
        producer_name=onx.producer_name,
        producer_version=onx.producer_version,
        model_version=onx.model_version,
        doc_string=onx.doc_string,
        opset_imports=opsets,
        domain=onx.domain,
    )
    return new_model
