import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import numpy as np
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
)
from onnx.compose import merge_models
from onnx.helper import (
    make_attribute,
    make_node,
    make_graph,
    make_function,
    make_model,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
    set_model_props,
)
from onnx.numpy_helper import from_array, to_array
from onnx.shape_inference import infer_shapes as onnx_infer_shapes
from onnx.version_converter import convert_version
from .onnx_io import load_model

logger = logging.getLogger("onnx-extended")

_rev_type: Dict[int, str] = {
    getattr(TensorProto, k): k
    for k in dir(TensorProto)
    if isinstance(getattr(TensorProto, k), int)
}


def _make_att_graph(name: str, new_body: GraphProto) -> AttributeProto:
    attr = AttributeProto()
    attr.name = name
    attr.g.CopyFrom(new_body)
    attr.type = AttributeProto.GRAPH
    return attr


def _apply_optimisation_on_graph(
    fct: Callable,
    onnx_model: Union[ModelProto, GraphProto, FunctionProto],
    recursive: bool = True,
    debug_info: Optional[List[str]] = None,
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
        graph = fct(onnx_model.graph, debug_info=[*debug_info, "GRAPH"], **kwargs)
        functions = [
            fct(
                f,
                debug_info=[*debug_info, f"FUNCTION {f.name}"],
                recursive=recursive,
                **kwargs,
            )
            for f in onnx_model.functions
        ]
        if hasattr(onnx_model, "value_info"):
            graph.value_info.extend(onnx_model.value_info)
        new_model = make_model(
            graph,
            opset_imports=[
                make_opsetid(d.domain, d.version) for d in onnx_model.opset_import
            ],
            functions=functions,
        )
        new_model.ir_version = onnx_model.ir_version
        new_model.producer_name = onnx_model.producer_name
        new_model.producer_version = onnx_model.producer_version
        new_model.domain = onnx_model.domain
        new_model.model_version = onnx_model.model_version
        new_model.doc_string = onnx_model.doc_string
        return new_model
    raise TypeError(
        f"This function only works on 'ModelProto' anod not not on {type(onnx_model)}."
    )


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
        debug_info.extend(str(type(onnx_model)).rsplit(".", maxsplit=1)[-1].strip("'>"))

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

    # mark outputs
    if is_function:
        marked = {o: set() for o in graph.output}
    else:
        marked = {o.name: set() for o in graph.output}
    nodes = list(graph.node)

    # Handles subgraphs.
    sub_graphs = []
    for node in nodes:
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                sub_graphs.append(node)
                break

    if sub_graphs:
        raise NotImplementedError(
            "Remove unused is not implemented when there are subgraphs."
        )

    # mark node output
    for node in reversed(nodes):
        used = False
        for o in node.output:
            if o in marked:
                for i in node.input:
                    marked[o].add(i)
                    used = True
        if used:
            for i in node.input:
                marked[i] = set()

    # removed nodes
    removed = set()
    marked_set = set(marked)
    for ind, node in enumerate(nodes):
        if not (set(node.output) & marked_set):
            removed.add(ind)

    if not is_function:
        initializers = [i for i in graph.initializer if i.name in marked]
        sparse_initializers = [i for i in graph.sparse_initializer if i.name in marked]
    new_nodes = [node for i, node in enumerate(nodes) if i not in removed]

    if sub_graphs and recursive:
        raise NotImplementedError(
            "Remove unused is not implemented when there are subgraphs."
        )

    # Finally create the new graph.
    if is_function:
        logger.debug("onnx_remove_node_unused:end function with %d nodes.", len(nodes))
        return make_function(
            onnx_model.domain,
            onnx_model.name,
            onnx_model.input,
            onnx_model.output,
            new_nodes,
            opset_imports=onnx_model.opset_import,
            attributes=onnx_model.attribute,
            doc_string=onnx_model.doc_string,
        )
    graph = make_graph(
        new_nodes,
        onnx_model.name,
        onnx_model.input,
        onnx_model.output,
        initializers,
        sparse_initializers,
    )
    graph.value_info.extend(onnx_model.value_info)
    logger.debug("onnx_remove_node_unused:end graph with %d nodes.", len(nodes))
    return graph


def _guess_proto_dtype(dtype) -> int:
    return np_dtype_to_tensor_dtype(dtype)


def get_tensor_shape(
    obj: Union[ValueInfoProto, TypeProto, TensorProto],
) -> Optional[List[Union[int, str, None]]]:
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
    if not shape:
        return shape
    return [None if s in (0, "") else s for s in shape]


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
            inits = set(i.name for i in att.g.initializer)
            inputs |= hidden - (inits & hidden)
    return inputs - (outputs & inputs)


def enumerate_model_node_outputs(
    model: ModelProto, add_node: bool = False, order: bool = False
) -> Iterable[Union[str, Tuple[str, NodeProto]]]:
    """
    Enumerates all the nodes of a model.

    :param model: :epkg:`ONNX` graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :param order: goes through outputs following the graph order
    :return: enumerator
    """
    assert hasattr(
        model, "graph"
    ), "Parameter model is not an ONNX model but {type(model)}"
    if order:
        edges = []
        dorder = {}
        node_names = {}
        for inp in model.graph.input:
            dorder[0, inp.name] = 0
        for node in model.graph.node:
            dorder[1, node.name] = 0
            for i in node.input:
                edges.append(("in", i, node.name))
            for o in node.output:
                edges.append(("out", o, node.name))
                node_names[o] = node
                dorder[0, o] = 0

        modif = 1
        n_iter = 0
        while modif > 0 and n_iter <= len(model.graph.node):
            modif = 0
            n_iter += 1
            for kind, data_name, node_name in edges:
                if kind == "in":
                    if (0, data_name) not in dorder:
                        continue
                    if dorder[0, data_name] + 1 > dorder[1, node_name]:
                        modif += 1
                        dorder[1, node_name] = dorder[0, data_name] + 1
                else:
                    if dorder[1, node_name] + 1 > dorder[0, data_name]:
                        modif += 1
                        dorder[0, data_name] = dorder[1, node_name] + 1

        orders = [(v, k) for k, v in dorder.items()]
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
        from onnx_extended.tools.onnx_nodes import select_model_inputs_outputs

        onx = onnx.load(path)
        onx2 = select_model_inputs_outputs(
            onx, inputs=["a", "b"],
            infer_shapes=True, verbose=1,
            overwrite={'a': (numpy.int32, None), 'b': (numpy.int64, None)})
        onnx.save(onx2, path2)
    """
    if not isinstance(model, ModelProto):
        raise TypeError(f"Unexpected type {type(model)} for model.")
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
        assert out in mark_var, "Output '{out}' not found in model."
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
                "[select_model_inputs_outputs] %s %s (%s) -> %s [%s]",
                s,
                node.op_type,
                ", ".join(node.input),
                ", ".join(node.output),
                node.name,
            )

    known_shapes = {}
    if infer_shapes:
        shapes = onnx_infer_shapes(model)
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
            "[select_model_inputs_outputs] nodes %r --> %r",
            len(model.graph.node),
            len(keep_nodes),
        )
        logger.info(
            "[select_model_inputs_outputs] inputs: %r", [_.name for _ in var_in]
        )
        logger.info(
            "[select_model_inputs_outputs] inputs: %r", [_.name for _ in var_out]
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
    if model.metadata_props:
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


def _info_type(typ: Union[TensorProto, TypeProto, SparseTensorProto]) -> Dict[str, str]:
    if typ is None:
        return {}
    if isinstance(typ, (TensorProto, SparseTensorProto)):
        shape = [str(i) for i in typ.dims]
        return dict(
            type="tensor", elem_type=_rev_type[typ.data_type], shape="x".join(shape)
        )
    if typ.tensor_type:
        ret = dict(type="tensor", elem_type=_rev_type[typ.tensor_type.elem_type])
        shape = []
        for d in typ.tensor_type.shape.dim:
            if d.dim_value:
                shape.append(str(d.dim_value))
            else:
                shape.append(d.dim_param or "?")
        ret["shape"] = "x".join(shape)
        return ret

    return dict(kind=str(type(typ)))


def enumerate_onnx_node_types(
    model: Union[str, ModelProto, GraphProto],
    level: int = 0,
    shapes: Optional[Dict[str, TypeProto]] = None,
    external: bool = True,
) -> Iterable[Dict[str, Union[str, float]]]:
    """
    Looks into types for every node in a model.

    :param model: a string or a proto
    :param level: level (recursivity level)
    :param shapes: known shapes,
        returned by :func:onnx.shape_inference.infer_shapes`
    :param external: loads the external data if the model is loaded
    :return: a list of dictionary which can be turned into a dataframe.
    """
    proto = load_model(model, external=external)
    if shapes is None and isinstance(proto, ModelProto):
        p2 = onnx_infer_shapes(proto)
        values = p2.graph.value_info
        shapes = {}
        for value in values:
            shapes[value.name] = value.type
        for o in proto.graph.output:
            if o.name not in shapes:
                shapes[o.name] = o.type

    if isinstance(proto, ModelProto):
        if shapes is None:
            raise RuntimeError("shape inference has failed.")
        for item in enumerate_onnx_node_types(proto.graph, level=level, shapes=shapes):
            yield item

    elif isinstance(model, FunctionProto):
        raise NotImplementedError(f"Not implemented for type {type(proto)}.")

    else:
        for inp in proto.input:
            obs = dict(level=level, name=inp.name, kind="input")
            obs.update(_info_type(inp.type))
            yield obs

        for init in proto.initializer:
            obs = dict(level=level, name=init.name, kind="initializer")
            obs.update(_info_type(init))
            yield obs

        for init in proto.sparse_initializer:
            obs = dict(level=level, name=init.name, kind="sparse_initializer")
            obs.update(_info_type(init))
            yield obs

        for node in proto.node:
            obs = dict(
                level=level,
                name=node.name,
                kind="Op",
                domain=node.domain,
                type=node.op_type,
                inputs=",".join(node.input),
                outputs=",".join(node.output),
                input_types=",".join(
                    _info_type(shapes.get(i, None)).get("elem_type", "")
                    for i in node.input
                ),
                output_types=",".join(
                    _info_type(shapes.get(i, None)).get("elem_type", "")
                    for i in node.output
                ),
            )
            yield obs

            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    obs = dict(name=att.name, kind="attribute", level=level + 1)
                    yield obs
                    for item in enumerate_onnx_node_types(
                        att.g, level=level + 1, shapes=shapes
                    ):
                        yield item

            for out in node.output:
                obs = dict(name=out, kind="result", level=level)
                obs.update(_info_type(shapes.get(out, None)))
                yield obs

        for out in proto.output:
            obs = dict(level=level, name=out.name, kind="output")
            obs.update(_info_type(out.type))
            yield obs


def enumerate_model_tensors(
    model: ModelProto,
) -> Iterable[Tuple[TensorProto, bool]]:
    """
    Enumerates all tensors in a model.

    :param model: model to process
    :return: iterator on a couple (TensorProto, bool),
        the boolean indicates if the data is external
    """
    from onnx.external_data_helper import (
        _get_all_tensors,
        uses_external_data,
    )

    for tensor in _get_all_tensors(model):
        yield tensor, uses_external_data(tensor)


def onnx_merge_models(
    m1: ModelProto, m2: ModelProto, io_map: List[Tuple[str, str]], verbose: int = 0
) -> ModelProto:
    """
    Merges two models. The functions also checks that the model
    have the same defined opsets (except for function).
    If not, the most recent opset is selected.

    :param m1: first model
    :param m2: second model
    :param io_map: mapping between outputs of the first model and
        and the input of the second one
    :param verbose: display some information if one of the model was updated
    :return: new model
    """
    opsets1 = {o.domain: o.version for o in m1.opset_import}
    opsets2 = {o.domain: o.version for o in m2.opset_import}
    update = {}
    for k, v2 in opsets2.items():
        if k not in opsets1:
            continue
        v1 = opsets1[k]
        if v1 == v2:
            continue
        update[k] = max(v1, v2)
    if update and verbose > 0:
        print(f"[onnx_merge_models] selected opsets: {update}")
    if "" in update:
        if opsets1[""] != update[""]:
            if verbose:
                print(
                    f"[onnx_merge_models] update model 1 from "
                    f"{opsets1['']} to {update['']}"
                )
            m1 = convert_version(m1, update[""])
        if opsets2[""] != update[""]:
            if verbose:
                print(
                    f"[onnx_merge_models] update model 2 from "
                    f"{opsets2['']} to {update['']}"
                )
            m2 = convert_version(m2, update[""])
    if m1.ir_version != m2.ir_version:
        new_ir = max(m1.ir_version, m2.ir_version)
        m1.ir_version = new_ir
        m2.ir_version = new_ir
    if verbose:
        for k, v in update.items():
            if k == "":
                continue
            print(f"[onnx_merge_models] no update implemented for domain {k!r} to {v}")
    return merge_models(m1, m2, io_map=io_map)


def tree_ensemble_use_as_tensor_attributes(node: NodeProto) -> NodeProto:
    """
    Uses only attributes suffixed with `_as_tensor` for tree ensemble operators.

    :param node: node to update
    :return: modified node
    """
    if node.op_type in {"TreeEnsembleRegressor", "TreeEnsembleClassifier"}:
        atts = {
            "base_values",
            "nodes_hitrates",
            "nodes_values",
            "target_weights",
            "class_weights",
        }
        attributes = []
        modified = False
        for att in node.attribute:
            if att.name not in atts:
                attributes.append(att)
                continue
            floats = list(att.floats)
            tensor = make_tensor(att.name, TensorProto.FLOAT, [len(floats)], floats)
            att = make_attribute(att.name + "_as_tensor", tensor)
            attributes.append(att)
            modified = True
        if not modified:
            return node
        new_node = make_node(
            node.op_type, node.input, node.output, name=node.name, domain=node.domain
        )
        new_node.attribute.extend(attributes)
        return new_node

    raise NotImplementedError(
        f"Unable to apply tree_ensemble_use_as_tensor_attributes "
        f"on operator type {node.op_type}."
    )


def convert_onnx_model(
    onnx_model: Union[ModelProto, GraphProto, NodeProto, FunctionProto],
    opsets: Dict[str, int],
    recursive: bool = True,
    use_as_tensor_attributes: bool = True,
    verbose: int = 0,
    _from_opset: Optional[Dict[str, int]] = None,
    debug_info: Optional[List[str]] = None,
) -> Union[ModelProto, GraphProto, NodeProto, FunctionProto]:
    """
    Upgrades a model to the latest opsets.

    :param onnx_model: proto
    :param opsets: list of opsets to update
    :param recursive: looks into subgraphs
    :param use_as_tensor_attributes: use attributes siffixed with `as_tensor` for trees
    :param verbose: verbosity
    :param _from_opset: tells which opset a node belongs too, only used when
        onnx_model is a NodeProto
    :param debug_info: unused
    :return: new proto
    """

    def _change_opsets(proto):
        old_opsets = {d.domain: d.version for d in proto.opset_import}
        old_opsets.update(opsets)
        del proto.opset_import[:]
        for k, v in old_opsets.items():
            d = proto.opset_import.add()
            d.domain = k
            d.version = v

    if isinstance(onnx_model, ModelProto):
        assert _from_opset is None, "_from_opset must be None for ModelProto."
        old_opsets = {d.domain: d.version for d in onnx_model.opset_import}
        if verbose > 0:
            print(
                f"[convert_onnx_model] upgrade ModelProto from opset "
                f"{old_opsets} to {opsets}"
            )
        new_opset = opsets.get("", old_opsets.get("", None))
        if new_opset != old_opsets.get("", None):
            onnx_model = convert_version(onnx_model, opsets)
        new_onnx = _apply_optimisation_on_graph(
            convert_onnx_model,
            onnx_model,
            recursive=recursive,
            verbose=verbose,
            opsets=opsets,
            use_as_tensor_attributes=use_as_tensor_attributes,
            _from_opset=old_opsets,
        )
        _change_opsets(new_onnx)
        return new_onnx

    is_function = isinstance(onnx_model, FunctionProto)
    if not is_function and not isinstance(_from_opset, dict):
        raise TypeError(f"_from_opset must a dictionary not {_from_opset!r}.")

    if isinstance(onnx_model, NodeProto):
        node = onnx_model
        if verbose > 1:
            print(
                f"[convert_onnx_model] upgrade node {node.op_type!r} from opset "
                f"{_from_opset.get(node.domain, 0)} to {opsets.get(node.domain, 0)}"
            )

        if (
            _from_opset.get(node.domain, 0) == 0
            or node.domain != "ai.onnx.ml"
            or opsets.get(node.domain) is None
        ):
            return node
        if _from_opset[node.domain] != opsets[node.domain]:
            if node.op_type in {"TreeEnsembleRegressor", "TreeEnsembleClassifier"}:
                # nothing to do
                pass
            else:
                raise NotImplementedError(
                    f"No upgrade is available from {_from_opset} to "
                    f"{opsets[node.domain]} for operator type "
                    f"{node.domain}.{node.op_type}."
                )
        if node.op_type in {"TreeEnsembleRegressor", "TreeEnsembleClassifier"}:
            if use_as_tensor_attributes:
                if _from_opset[node.domain] < 3 and opsets.get(node.domain, 0) < 3:
                    raise RuntimeError(
                        "Opset 3 is required when use_as_tensor_attributes is True."
                    )
                new_node = tree_ensemble_use_as_tensor_attributes(node)
                if verbose > 2:
                    atts = ", ".join(a.name for a in new_node.attribute)
                    print(f"[convert_onnx_model] {node.op_type}: attributes={atts!r}")

                return new_node
        return node

    if is_function:
        old_opsets = {d.domain: d.version for d in onnx_model.opset_import}
    else:
        old_opsets = _from_opset

    if verbose > 1:
        print(
            f"[convert_onnx_model] upgrade {type(onnx_model)} from opset "
            f"{old_opsets} to {opsets}"
        )

    nodes = onnx_model.node
    new_nodes = []
    for node in nodes:
        new_nodes.append(
            convert_onnx_model(
                node,
                recursive=recursive,
                verbose=verbose,
                opsets=opsets,
                use_as_tensor_attributes=use_as_tensor_attributes,
                _from_opset=_from_opset,
            )
        )

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, new_nodes))
    if is_function:
        onx = make_function(
            onnx_model.domain,
            onnx_model.name,
            onnx_model.input,
            onnx_model.output,
            new_nodes,
            opset_imports=onnx_model.opset_import,  # opsets should have been updated
            attributes=onnx_model.attribute,
            doc_string=onnx_model.doc_string,
        )
        _change_opsets(onx)
        return onx

    graph = make_graph(
        new_nodes,
        onnx_model.name,
        onnx_model.input,
        onnx_model.output,
        onnx_model.initializer,
        sparse_initializer=onnx_model.sparse_initializer,
    )
    graph.value_info.extend(onnx_model.value_info)
    return graph


def multiply_tree(node: NodeProto, n: int, random: bool = True) -> NodeProto:
    """
    Multiplies the number of trees in TreeEnsemble operator.
    It replicates the existing trees but permutes features ids
    and node values if random is True.

    :param node: tree ensemble operator
    :param n: number of times the existing trees must be multiplied
    :param random: permutation or thresholds
    :return: the new trees
    """
    assert isinstance(node, NodeProto), f"node is not a NodeProto but {type(node)}."
    assert node.op_type.startswith(
        "TreeEnsemble"
    ), "Unexpected node type {node.op_type!r}."
    args = [node.op_type, node.input, node.output]
    kwargs = {"domain": node.domain}

    nodes_featureids = None
    not_leave_mask = None
    nodes_values = None
    for att in node.attribute:
        if att.name == "nodes_modes":
            not_leave_mask = np.array(att.strings) != "LEAF"
        elif att.name == "nodes_featureids":
            nodes_featureids = np.array(att.ints)
        elif att.name == "nodes_values":
            nodes_values = np.array(att.floats, dtype=np.float32)
        elif att.name == "nodes_values_as_tensor":
            nodes_values = to_array(att)
    assert not_leave_mask is not None, "Attribute nodes_modes is missing."
    assert nodes_featureids is not None, "Attribute nodes_featureids is missing."
    assert (
        nodes_values is not None
    ), "Attribute nodes_values or nodes_values_as_tensor is missing."

    # permutation
    new_nodes_values = []
    new_feature_ids = []
    indices = np.array(
        [i for i, m in zip(np.arange(len(nodes_featureids)), not_leave_mask) if m]
    )
    permuted_indices = indices.copy()
    for _i in range(n):
        new_feature_ids.extend(nodes_featureids.tolist())
        new_nodes_values.extend(nodes_values.tolist())
        if random:
            permuted_indices = np.random.permutation(permuted_indices)
            nodes_featureids[indices] = nodes_featureids[permuted_indices]
            nodes_values[indices] = nodes_values[permuted_indices]
    assert len(new_feature_ids) == len(
        new_nodes_values
    ), f"Dimension mismatch {len(nodes_featureids)} != {len(nodes_values)}"
    assert len(nodes_featureids) * n == len(
        new_nodes_values
    ), f"Dimension mismatch {len(nodes_featureids) * n} != {len(new_nodes_values)}"

    # other attributes
    for att in node.attribute:
        if att.name == "nodes_featureids":
            kwargs[att.name] = new_feature_ids
        elif att.name == "nodes_values":
            kwargs[att.name] = new_nodes_values
        elif att.name == "nodes_values_as_tensor":
            kwargs[att.name] = from_array(np.array(new_nodes_values, dtype=np.int64))
        elif att.name in {"aggregate_function", "post_transform"}:
            kwargs[att.name] = att.s
        elif att.name in {"n_targets"}:
            kwargs[att.name] = att.i
        elif att.name == "classlabels_int64s":
            ints = att.ints
            if ints:
                kwargs[att.name] = list(ints)
        elif att.name == "classlabels_strings":
            vals = att.strings
            if vals:
                kwargs[att.name] = list(vals)
        elif att.name == "base_values":
            fs = list(att.floats)
            if fs:
                kwargs.att[att.name] = fs
        elif att.name.endswith("_as_tensor") and att.name != "nodes_values_as_tensor":
            v = to_array(att.t)
            if att.name == "base_values_as_tensor":
                if v.shape:
                    kwargs[att.name] = from_array(v)
            else:
                kwargs[att.name] = from_array(np.repeat(v, n))
        elif att.name in {
            "class_ids",
            "class_nodeids",
            "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
            "nodes_nodeids",
            "nodes_truenodeids",
            "target_ids",
            "target_nodeids",
        }:
            kwargs[att.name] = list(att.ints) * n
        elif att.name in {"class_treeids", "nodes_treeids", "target_treeids"}:
            ints = []
            arr = np.array(att.ints, dtype=np.int64)
            for _i in range(n):
                ints.extend(arr.tolist())
                arr += 1
            kwargs[att.name] = ints
        elif att.name in {"nodes_modes"}:
            kwargs[att.name] = list(att.strings) * n
        elif att.name in {
            "nodes_hitrates",
            "target_weights",
            "class_weights",
        }:
            fs = list(att.floats)
            if fs:
                kwargs[att.name] = fs * n

    return make_node(*args, **kwargs)
