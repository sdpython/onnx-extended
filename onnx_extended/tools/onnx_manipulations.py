import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    ValueInfoProto,
    TensorProto,
    TypeProto,
    shape_inference,
)
from onnx.helper import (
    make_attribute,
    make_graph,
    make_function,
    make_model,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
    set_model_props,
)

logger = logging.getLogger("onnx-extended")


def _make_att_graph(name: str, new_body: GraphProto) -> AttributeProto:
    attr = AttributeProto()
    attr.name = name
    attr.g.CopyFrom(new_body)
    attr.type = AttributeProto.GRAPH
    return attr


def _make_node(
    op_type: str,
    inputs: List[str],
    outputs: List[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: str = "",
    attributes: Optional[Dict[str, Any]] = None,
) -> NodeProto:
    """
    Constructs a NodeProto.

    :param op_type: (string): The name of the operator to construct
    :param inputs: list of input names
    :param outputs: list of output names
    :param name: optional unique identifier for NodeProto
    :param doc_string: optional documentation
        string for NodeProto
    :param domain: optional domain for NodeProto.
        If it's None, we will just use default domain (which is empty)
    :param attributes: the attributes of the node. The acceptable values
        are documented in `make_attribute`.
    :return: node
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if isinstance(attributes, dict):
        if len(attributes) > 0:
            node.attribute.extend(
                make_attribute(key, value) for key, value in sorted(attributes.items())
            )
    elif attributes:
        for att in attributes:
            node.attribute.extend([att])
    return node


def _apply_optimisation_on_graph(
    fct: Callable,
    onnx_model: Union[ModelProto, GraphProto, FunctionProto],
    recursive: bool = True,
    debug_info: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> Union[ModelProto, GraphProto, FunctionProto]:
    """
    Applies an optimisation function *fct* on a graph
    and not on the model.

    :param fct: function to optimize like :func:`onnx_remove_node_unused`
    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param kwargs: additional parameters
    :return: new onnx _model
    """
    if hasattr(onnx_model, "graph"):
        if debug_info is None:
            debug_info = []
        graph = fct(onnx_model.graph, debug_info=debug_info + ["GRAPH"], **kwargs)
        new_model = make_model(graph, functions=onnx_model.functions)
        new_model.ir_version = onnx_model.ir_version
        new_model.producer_name = onnx_model.producer_name
        new_model.producer_version = onnx_model.producer_version
        new_model.domain = onnx_model.domain
        new_model.model_version = onnx_model.model_version
        new_model.doc_string = onnx_model.doc_string
        if hasattr(onnx_model, "value_info"):
            graph.value_info.extend(onnx_model.value_info)
        while len(new_model.opset_import) > 0:
            new_model.opset_import.pop()
        for oimp in onnx_model.opset_import:
            op_set = new_model.opset_import.add()
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return new_model
    raise TypeError(
        f"This function only works on 'ModelProto' anod not not on {type(onnx_model)}."
    )


def _apply_remove_node_fct_node(
    fct: Callable, node: NodeProto, recursive: bool, debug_info: str
) -> NodeProto:
    """
    Applies an optimizing function on a subgraphs.

    :param node: onnx node
    :param recursive: does it in subgraphs as well
    :return: new node
    """
    if not hasattr(node, "attribute"):
        return node
    modified = 0
    new_atts = []
    for att in node.attribute:
        if att.name in ("body", "then_branch", "else_branch"):
            new_body = fct(
                att.g, recursive=recursive, debug_info=debug_info + [att.name]
            )
            new_atts.append(_make_att_graph(att.name, new_body))
            modified += 1
        else:
            new_atts.append(att)
    if modified > 0:
        new_node = _make_node(
            node.op_type, node.input, node.output, name=node.name, attributes=new_atts
        )
        return new_node
    return node


def _process_node(
    node: NodeProto,
    data: Dict,
    edges: Dict,
    paths: Dict,
    prefix: str = "",
    sep: str = ":X:",
    path: Optional[List[str]] = None,
):
    node_name = prefix + node.name
    data[node_name, 1] = node
    path = [] if path is None else path.copy()
    paths[node_name, 1] = path
    path = path.copy()
    path.append(node_name)
    for inp in node.input:
        data[inp, 0] = node
        edges[(inp, 0), (node_name, 1)] = node
        paths[inp, 0] = path
        if sep in node_name:
            # We need to link an input to the parent node
            # if the node is part of subgraph.
            # path_r = paths[inp, 0]
            if len(path) <= 1:
                raise RuntimeError(
                    f"Unexpected path {path!r}, this may happen "
                    f"if sep={sep!r} is already used in the original model."
                )
            edges[(inp, 0), (path[-2], 1)] = node

    for out in node.output:
        data[out, 0] = node
        paths[out, 0] = node_name
        edges[(node_name, 1), (out, 0)] = node
    if len(node.attribute) > 0:
        for att in node.attribute:
            if not hasattr(att, "g"):
                continue
            if not isinstance(att.g, GraphProto):
                continue
            for no in att.g.node:
                _process_node(no, data, edges, paths, prefix=node_name + sep, path=path)


def onnx_remove_node_unused(onnx_model, recursive=True, debug_info=None, **options):
    """
    Removes unused nodes of the graph. An unused node
    is not involved in the output computation.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param options: unused
    :return: new onnx _model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).rsplit(".", maxsplit=1)[-1].strip("'>")]
    else:
        debug_info = debug_info + [
            str(type(onnx_model)).rsplit(".", maxsplit=1)[-1].strip("'>")
        ]

    if hasattr(onnx_model, "graph"):
        return _apply_optimisation_on_graph(
            onnx_remove_node_unused,
            onnx_model,
            recursive=recursive,
            debug_info=debug_info,
            **options,
        )

    graph = onnx_model
    logger.debug("onnx_remove_node_unused:begin with %d nodes.", len(graph.node))
    is_function = isinstance(graph, FunctionProto)
    data = {}
    valid = {}
    edges = {}
    paths = {}

    if not is_function:
        for init in graph.initializer:
            data[init.name, 0] = init

    for node in graph.node:
        _process_node(node, data, edges, paths)

    for out in graph.output:
        valid[out if is_function else out.name, 0] = True

    modif = 1
    while modif > 0:
        modif = 0
        for e1, e2 in edges:  # pylint: disable=E1141
            if valid.get(e2, False) and not valid.get(e1, False):
                valid[e1] = True
                modif += 1

    new_nodes = [n for n in graph.node if (n.name, 1) in valid]
    if not is_function:
        new_inits = [n for n in graph.initializer if (n.name, 0) in valid]

    if recursive:
        # Handles subgraphs.
        for i in range(len(new_nodes)):
            node = new_nodes[i]
            if node is None or not (node.attribute):
                continue
            new_nodes[i] = _apply_remove_node_fct_node(
                onnx_remove_node_unused,
                node,
                recursive=True,
                debug_info=debug_info + [node.name],
            )

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, new_nodes))
    if is_function:
        logger.debug("onnx_remove_node_unused:end function with %d nodes.", len(nodes))
        return make_function(
            onnx_model.domain,
            onnx_model.name,
            onnx_model.input,
            onnx_model.output,
            nodes,
            opset_imports=onnx_model.opset_import,
            attributes=onnx_model.attribute,
            doc_string=onnx_model.doc_string,
        )
    graph = make_graph(
        nodes, onnx_model.name, onnx_model.input, onnx_model.output, new_inits
    )
    graph.value_info.extend(onnx_model.value_info)
    logger.debug("onnx_remove_node_unused:end graph with %d nodes.", len(nodes))
    return graph


def _guess_proto_dtype(dtype) -> int:
    return np_dtype_to_tensor_dtype(dtype)


def get_tensor_shape(
    obj: Union[ValueInfoProto, TypeProto, TensorProto]
) -> List[Union[int, str]]:
    """
    Returns the shape if that makes sense for this object.
    """
    if isinstance(obj, ValueInfoProto):
        return get_tensor_shape(obj.type)
    elif not isinstance(obj, TypeProto):
        raise TypeError(f"Unexpected type {type(obj)!r}.")
    if not obj.tensor_type.HasField("shape"):
        return None
    shape = []
    for d in obj.tensor_type.shape.dim:
        v = d.dim_value if d.dim_value > 0 else d.dim_param
        shape.append(v)
    if len(shape) == 0:
        return shape
    return list(None if s in (0, "") else s for s in shape)


def get_hidden_inputs(nodes: Iterable[NodeProto]) -> Set[str]:
    """
    Returns the list of hidden inputs used by subgraphs.

    :param nodes: list of nodes
    :return: list of names
    """
    inputs = set()
    outputs = set()
    for node in nodes:
        inputs |= set(node.input)
        outputs |= set(node.output)
        for att in node.attribute:
            if (
                att.type != AttributeProto.GRAPH
                or not hasattr(att, "g")
                or att.g is None
            ):
                continue
            hidden = get_hidden_inputs(att.g.node)
            inits = set([i.name for i in att.g.initializer])
            inputs |= hidden - (inits & hidden)
    return inputs - (outputs & inputs)


def enumerate_model_node_outputs(
    model: ModelProto, add_node: bool = False, order: bool = False
) -> Iterable:
    """
    Enumerates all the nodes of a model.

    :param model: :epkg:`ONNX` graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :param order: goes through outputs following the graph order
    :return: enumerator
    """
    if not hasattr(model, "graph"):
        raise TypeError(f"Parameter model is not an ONNX model but {type(model)}")
    if order:
        edges = []
        order = {}
        node_names = {}
        for inp in model.graph.input:
            order[0, inp.name] = 0
        for node in model.graph.node:
            order[1, node.name] = 0
            for i in node.input:
                edges.append(("in", i, node.name))
            for o in node.output:
                edges.append(("out", o, node.name))
                node_names[o] = node
                order[0, o] = 0

        modif = 1
        n_iter = 0
        while modif > 0 and n_iter <= len(model.graph.node):
            modif = 0
            n_iter += 1
            for kind, data_name, node_name in edges:
                if kind == "in":
                    if (0, data_name) not in order:
                        continue
                    if order[0, data_name] + 1 > order[1, node_name]:
                        modif += 1
                        order[1, node_name] = order[0, data_name] + 1
                else:
                    if order[1, node_name] + 1 > order[0, data_name]:
                        modif += 1
                        order[0, data_name] = order[1, node_name] + 1

        orders = [(v, k) for k, v in order.items()]
        orders.sort()

        for _, k in orders:
            if k[0] == 1:
                continue
            out = k[1]
            if out not in node_names:
                continue
            yield (out, node_names[out]) if add_node else out
    else:
        for node in model.graph.node:
            for out in node.output:
                yield (out, node) if add_node else out


def select_model_inputs_outputs(
    model: ModelProto,
    outputs: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
    infer_shapes: bool = True,
    overwrite: Optional[Dict[str, Any]] = None,
    remove_unused: bool = True,
    verbose: int = 0,
):
    """
    Takes a model and changes its outputs.

    :param model: :epkg:`ONNX` model
    :param inputs: new inputs, same ones if None
    :param outputs: new outputs, same ones if None
    :param infer_shapes: infer inputs and outputs shapes
    :param overwrite: overwrite type and shapes for
        inputs or outputs, *overwrite* is a
        dictionary `{'name': (numpy dtype, shape)}`
    :param remove_unused: remove unused nodes from the graph
    :param verbose: display information while converting
    :return: modified model

    The function removes unneeded nodes.

    The following example shows how to change the inputs of model
    to bypass the first nodes. Shape inferences fails to determine
    the new inputs type. They need to be overwritten.
    `verbose=1` shows the number of deleted nodes.

    ::

        import onnx
        from onnx_extended.tools.onnx_manipulations import select_model_inputs_outputs

        onx = onnx.load(path)
        onx2 = select_model_inputs_outputs(
            onx, inputs=["a", "b"],
            infer_shapes=True, verbose=1,
            overwrite={'a': (numpy.int32, None), 'b': (numpy.int64, None)})
        onnx.save(onx2, path2)
    """
    if inputs is not None and not isinstance(inputs, list):
        inputs = [inputs]
    if outputs is not None and not isinstance(outputs, list):
        outputs = [outputs]
    if inputs is None:
        inputs = [i.name for i in model.graph.input]
    if outputs is None:
        outputs = [o.name for o in model.graph.output]

    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in inputs:
        mark_var[inp] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError(f"Output '{out}' not found in model.")
        mark_var[out] = 1

    nodes = list(model.graph.node[::-1])
    mark_op = {}
    for node in list(nodes):
        mark_op[id(node)] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[id(node)] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[id(node)] = 1
                    mod = True
                    break
            if not mod:
                continue

            hidden = get_hidden_inputs([node])
            node_inputs = list(node.input) + list(hidden)

            nb += 1
            for inp in node_inputs:
                if inp in inputs:
                    continue
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes[::-1] if mark_op[id(node)] == 1]

    if verbose > 1:
        for node in nodes:
            s = "+" if mark_op[id(node)] == 1 else "-"
            logger.info(
                "[select_model_inputs_outputs] %s %s (%s) -> %s [%s]"
                % (
                    s,
                    node.op_type,
                    ", ".join(node.input),
                    ", ".join(node.output),
                    node.name,
                )
            )

    known_shapes = {}
    if infer_shapes:
        shapes = shape_inference.infer_shapes(model)
        for shape in shapes.graph.value_info:
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.input:
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.output:
            known_shapes[shape.name] = shape.type
    else:
        for shape in model.graph.input:
            known_shapes[shape.name] = shape.type
        for shape in model.graph.output:
            known_shapes[shape.name] = shape.type

    var_in = []
    for name in inputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = _guess_proto_dtype(dtype)
            value_info = make_tensor_value_info(name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = make_tensor_value_info(name, proto_dtype, shape)
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_in.append(value_info)

    var_out = []
    for name in outputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = _guess_proto_dtype(dtype)
            value_info = make_tensor_value_info(name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = make_tensor_value_info(name, proto_dtype, shape)
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_out.append(value_info)

    if verbose > 0:
        logger.info(
            "[select_model_inputs_outputs] nodes %r --> %r"
            % (len(model.graph.node), len(keep_nodes))
        )
        logger.info(
            "[select_model_inputs_outputs] inputs: %r" % [_.name for _ in var_in]
        )
        logger.info(
            "[select_model_inputs_outputs] inputs: %r" % [_.name for _ in var_out]
        )

    graph = make_graph(
        keep_nodes,
        model.graph.name,
        var_in,
        var_out,
        model.graph.initializer,
        sparse_initializer=model.graph.sparse_initializer,
    )
    onnx_model = make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    # remove unused nodes
    if remove_unused:
        onnx_model = onnx_remove_node_unused(onnx_model, recursive=False)

    return onnx_model
