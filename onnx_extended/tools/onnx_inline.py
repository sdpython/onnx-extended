import pprint
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    ValueInfoProto,
)
from onnx.helper import (
    make_graph,
    make_function,
    make_model,
    make_node,
    make_operatorsetid,
    make_attribute,
    make_value_info,
)


def enumerate_onnx_names(
    onx: Union[FunctionProto, GraphProto, ModelProto],
) -> Iterator[str]:
    """
    Enumerates all existing names in one ONNX graph
    (ModelProto, FunctionProto, GraphProto).
    The function is recursive.

    :param onx: one onnx object
    :return: iterator on names
    """
    if hasattr(onx, "graph"):
        for i in onx.graph.initializer:
            yield i.name
        for i in onx.graph.input:
            yield i.name
        for i in onx.graph.output:
            yield i.name
        nodes = onx.graph.node
    elif hasattr(onx, "initializer"):
        for i in onx.initializer:
            yield i.name
        for i in onx.input:
            yield i.name
        for i in onx.output:
            yield i.name
        nodes = onx.node
    else:
        if hasattr(onx, "input"):
            yield from onx.input
        if hasattr(onx, "output"):
            yield from onx.output
        nodes = onx.node
    for node in nodes:
        yield from node.input
        yield from node.output
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                yield from enumerate_onnx_names(att.g)


def enumerate_onnx_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto],
) -> Iterator[str]:
    """
    Enumerates all nodes in one ONNX graph
    (ModelProto, FunctionProto, GraphProto).
    The function is recursive.

    :param onx: one onnx object
    :return: iterator on names
    """
    if isinstance(onx, list):
        nodes = onx
    elif hasattr(onx, "graph"):
        nodes = onx.graph.node
    else:
        nodes = onx.node
    for node in nodes:
        yield node
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                yield from enumerate_onnx_nodes(att.g)


def _get_new_name(
    prefix: str, name: Union[str, ValueInfoProto], existing_names: Set[str]
) -> str:
    opt = f"{prefix}_{name}_0"
    i = 0
    while opt in existing_names:
        i += 1
        opt = "%s_%s_%d" % (prefix, name, i)
    existing_names.add(opt)
    return opt


def onnx_subgraphs_level(obj: Union[FunctionProto, GraphProto, ModelProto]) -> int:
    """
    Returns the depth of the graph.

    :param obj: onnx object
    :return: integer
    """
    if isinstance(obj, ModelProto):
        return onnx_subgraphs_level(obj.graph)
    best = 0
    for node in obj.node:
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                m = onnx_subgraphs_level(att.g)
                if m > best:
                    best = m
    return best + 1


class _inline_mapping(dict):
    """
    Overwrites class dictionary to debug more easily.

    :param verbose: verbosity
    :param level: sub graph level
    """

    def __init__(self, verbose: int, level: int):
        dict.__init__(self)
        self._verbose = verbose
        self._level = level

    def __setitem__(self, key: str, value: Any):
        "Adds a value."
        if self._verbose > 3:
            print(
                "[_inline_mapping-dict-addkv] %s + %r: %r"
                % ("  " * self._level, key, value)
            )
        assert key not in self, (
            "Key %r was already added (with value %r, new one is %r)."
            ""
            % (  # noqa: ISC001
                key,
                self[key],
                value,
            )
        )
        dict.__setitem__(self, key, value)

    def update(self, d: Dict[str, Any]):
        "Updates many values."
        for k, v in d.items():
            self[k] = v

    def copy(self) -> Dict[str, Any]:
        "Returns a copy."
        m = _inline_mapping(self._verbose, self._level)
        for k, v in self.items():
            m[k] = v
        return m

    def remove(self, o: str):
        "Removes one element."
        assert o in self, f"Cannot remove a key {o!r}."
        self.pop(o)


def _onnx_inline_function_graph(
    graph: GraphProto,
    protos: Dict[str, FunctionProto],
    existing_names: Set[str],
    mapping: _inline_mapping,
    verbose: int,
    rename: bool,
    level: int,
) -> Tuple[
    Union[FunctionProto, GraphProto, ModelProto], List[Union[ValueInfoProto, NodeProto]]
]:
    if len(graph.node) == 0:
        # Outputs have still to be renamed.
        graph0 = graph
        if verbose > 1:
            print(
                "[onnx_inline_function-graph] %s visit0 graph=%d rename=%r "
                "len(mapping)=%d begin"
                % ("  " * level, id(graph), rename, len(mapping))
            )
        if rename:
            modified_nodes = []
            mapping = mapping.copy()
            for i in graph.input:
                mapping[i.name] = i.name
            for i in graph.initializer:
                mapping[i.name] = i.name
            for i in graph.sparse_initializer:
                mapping[i.name] = i.name
            outputs = []
            for o in graph.output:
                no = make_value_info(mapping[o.name], o.type)
                if no.name != o.name:
                    modified_nodes.append(o)
                    outputs.append(no)
                else:
                    outputs.append(o)
            if len(modified_nodes) > 0:
                graph = make_graph(
                    [],
                    graph.name,
                    graph.input,
                    outputs,
                    graph.initializer,
                    doc_string=graph.doc_string,
                    sparse_initializer=list(graph.sparse_initializer),
                )
        else:
            modified_nodes = []

        if verbose > 1:
            print(
                "[onnx_inline_function-graph] %s visit graph=%d end "
                "changed=%r len(modified_nodes)=%d"
                % (
                    "  " * level,
                    id(graph0),
                    id(graph0) != id(graph),
                    len(modified_nodes),
                )
            )

        return graph, modified_nodes

    graph0 = graph
    mapping = mapping.copy()
    init = list(graph.initializer)
    init_sparse = list(graph.sparse_initializer)
    inputs = list(graph.input)
    modified_nodes = []
    outputs = list(graph.output)

    if verbose > 1:
        print(
            "[onnx_inline_function-graph] %s >visit graph=%d rename=%r "
            "len(mapping)=%d begin" % ("  " * level, id(graph), rename, len(mapping))
        )

    output_names = [o.name for o in outputs]
    for i in init:
        mapping[i.name] = i.name
    for i in init_sparse:
        mapping[i.name] = i.name
    for i in inputs:
        mapping[i.name] = i.name

    # first step, replace names
    nodes = []
    for node in list(graph.node):
        mod = 0
        inp = []
        for i in node.input:
            assert (
                i in mapping
            ), "Cannot find input %r in %s for node (level=%d)\n%r." % (
                i,
                pprint.pformat(mapping),
                level,
                node,
            )
            inp.append(mapping[i])
            if mapping[i] != i:
                mod += 1
        out = []
        for o in node.output:
            new_o = o
            if rename:
                if o not in output_names:
                    new_o = _get_new_name("_inl", o, existing_names)
                if o in mapping:
                    # See below.
                    mapping.remove(o)
            elif o in mapping:
                # That means the main contains a result node but is overwritten by
                # the subgraph. The local variable cannot be reached anymore,
                # we remove it.
                mapping.remove(o)
                if o in node.input:
                    new_o = _get_new_name("_inl", o, existing_names)
                if verbose > 3:
                    print(
                        "[onnx_inline_function-renam] %s node %r(%r): %r -> %r "
                        "overwrite result (%r -> %r)."
                        % (
                            "  " * level,
                            node.op_type,
                            node.name,
                            node.input,
                            node.output,
                            o,
                            new_o,
                        )
                    )
            out.append(new_o)
            mapping[o] = new_o
            if o != new_o:
                mapping[new_o] = new_o
                mod += 1

        if verbose > 3:
            print(
                "[onnx_inline_function-renam] %s rep node %r(%r): %r -> %r"
                % ("  " * level, node.op_type, node.name, node.input, node.output)
            )
        new_node = make_node(
            node.op_type,
            inp,
            out,
            domain=node.domain,
            name=_get_new_name("_inln", node.name, existing_names),
        )
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                g, m = _onnx_inline_function_graph(
                    att.g,
                    protos,
                    existing_names=existing_names,
                    verbose=verbose,
                    mapping=mapping,
                    rename=rename,
                    level=level + 1,
                )
                if len(m) > 0:
                    att = make_attribute(att.name, g)
                    mod += len(m)
                else:
                    att = make_attribute(att.name, att.g)
            new_node.attribute.append(att)
        if mod > 0:
            if verbose > 2:
                print(
                    "[onnx_inline_function-renam] %s add node %r(%r): %r -> %r"
                    % (
                        "  " * level,
                        new_node.op_type,
                        new_node.name,
                        new_node.input,
                        new_node.output,
                    )
                )
            nodes.append(new_node)
            modified_nodes.append(node)
        else:
            nodes.append(node)

    if len(modified_nodes) > 0:
        if verbose > 1:
            print(
                "[onnx_inline_function-graph] %s -1 graph=%d "
                "len(modified_nodes)=%d"
                % ("  " * level, id(graph), len(modified_nodes))
            )

        graph = make_graph(
            nodes,
            graph.name,
            inputs,
            outputs,
            init,
            doc_string=graph.doc_string,
            sparse_initializer=list(graph.sparse_initializer),
        )
    elif not rename:
        # no modification, let's check the node hiding a functions
        new_nodes = []
        for node in nodes:
            nnodes, m = _onnx_inline_function_node(
                node, protos, existing_names, verbose, level=level
            )
            if len(m) > 0:
                if verbose > 0:
                    print(
                        "[onnx_inline_function-subgr] %s replaced node %r (%r) "
                        "with %d nodes (id=%r) -- %r -> %r"
                        % (
                            "  " * level,
                            node.name,
                            node.op_type,
                            len(nnodes),
                            id(node),
                            node.input,
                            node.output,
                        )
                    )
                new_nodes.extend(nnodes)
                modified_nodes.extend(m)
            else:
                new_nodes.append(node)
        if len(modified_nodes) > 0:
            if verbose > 1:
                print(
                    "[onnx_inline_function-graph] %s -2 graph=%d "
                    "len(modified_nodes)=%d"
                    % ("  " * level, id(graph), len(modified_nodes))
                )

            nodes = new_nodes
            graph = make_graph(
                nodes,
                graph.name,
                inputs,
                outputs,
                init,
                doc_string=graph.doc_string,
                sparse_initializer=list(graph.sparse_initializer),
            )

    if verbose > 1:
        print(
            "[onnx_inline_function-graph] %s <visit graph=%d end "
            "changed=%r len(modified_nodes)=%d"
            % ("  " * level, id(graph0), id(graph0) != id(graph), len(modified_nodes))
        )

    return graph, modified_nodes


def _onnx_inline_function_node(
    node: NodeProto,
    protos: Dict[str, FunctionProto],
    existing_names: Set[str],
    verbose: int,
    level: int,
) -> Tuple[List[NodeProto], List[Union[ValueInfoProto, NodeProto]]]:
    """
    Inline a node.

    :param node: node to inline
    :param protos: known functions
    :param existing_names: names which cannot be used
    :param verbose: verbosity level
    :param level: level of subgraphs
    :return: new nodes, modified nodes
    """
    # The function does not rename input or output
    # of the node, it just replaces the node but a function
    # if the function exists.
    modified_nodes = []
    key = node.domain, node.op_type
    if key in protos:
        proto = protos[key]
        assert isinstance(
            proto, FunctionProto
        ), "Prototype for key=%r must be a Function Proto, not %r." % (key, type(proto))
        modified_nodes.append(node)
        new_nodes = []
        mapping = _inline_mapping(verbose, level)
        prefix = "_inl"

        for fr, to in zip(node.input, proto.input):
            n = make_node("Identity", [fr], [_get_new_name(prefix, to, existing_names)])
            if verbose > 2:
                print(
                    "[onnx_inline_function-ninpu] %s add node %r(%r): %r -> %r"
                    % ("  " * level, n.op_type, n.name, n.input, n.output)
                )
            mapping[to] = n.output[0]
            if to != n.output[0]:
                mapping[n.output[0]] = n.output[0]
            new_nodes.append(n)

        attributes = {att.name: att for att in node.attribute}

        for nn in proto.node:
            new_input = [mapping[i] for i in nn.input]
            new_output = [_get_new_name(prefix, o, existing_names) for o in nn.output]
            mapping.update(dict(zip(nn.output, new_output)))
            mapping.update({oo: oo for oo in new_output})
            new_node = make_node(
                nn.op_type,
                new_input,
                new_output,
                domain=nn.domain,
                name=_get_new_name(prefix, nn.name, existing_names),
            )
            if verbose > 3:
                print(
                    "[onnx_inline_function-nnode]   %s rep node %r(%r): %r -> %r"
                    % ("  " * level, nn.op_type, nn.name, nn.input, nn.output)
                )
            if verbose > 2:
                print(
                    "[onnx_inline_function-nnode] %s add node %r(%r): %r -> %r"
                    % (
                        "  " * level,
                        new_node.op_type,
                        new_node.name,
                        new_node.input,
                        new_node.output,
                    )
                )
            for att in nn.attribute:
                if hasattr(att, "ref_attr_name") and att.ref_attr_name:
                    # linked attribute
                    assert att.ref_attr_name in attributes, (
                        f"A linked attribute {att.ref_attr_name!r} "
                        f"cannot be found in {list(sorted(attributes))} "
                        f"for operator type {nn.op_type!r} and attribute {att.name!r}."
                    )
                    new_att = AttributeProto()
                    new_att.ParseFromString(
                        attributes[att.ref_attr_name].SerializeToString()
                    )
                    new_att.name = att.name
                    att = new_att
                    if verbose > 3:
                        print(
                            "[onnx_inline_function-funct]   %s fct=%r att %r linked to %r"  # noqa: E501
                            % ("  " * level, key, att.name, att.ref_attr_name)
                        )
                elif (
                    att.type == AttributeProto.GRAPH
                    and hasattr(att, "g")
                    and att.g is not None
                ):
                    if verbose > 1:
                        print(
                            "[onnx_inline_function-funct] %s fct=%r graph=%d node=%d"
                            % ("  " * level, key, id(att.g), id(new_node))
                        )

                    g, m = _onnx_inline_function_graph(
                        att.g,
                        protos,
                        existing_names=existing_names,
                        verbose=verbose,
                        mapping=mapping,
                        rename=True,
                        level=level + 1,
                    )
                    if len(m) > 0:
                        att = make_attribute(att.name, g)
                    else:
                        att = make_attribute(att.name, att.g)
                new_node.attribute.append(att)
            new_nodes.append(new_node)

        for fr, to in zip(proto.output, node.output):
            n = make_node("Identity", [mapping[fr]], [to])
            if verbose > 2:
                print(
                    "[onnx_inline_function-noutt] %s add node %r(%r): %r -> %r"
                    % ("  " * level, n.op_type, n.name, n.input, n.output)
                )
            new_nodes.append(n)
    else:
        new_nodes = [node]
        modified_nodes = []
    return new_nodes, modified_nodes


def onnx_inline_function(
    obj: Union[FunctionProto, GraphProto, ModelProto],
    protos: Optional[Dict[str, Any]] = None,
    existing_names: Optional[Set[str]] = None,
    verbose: int = 0,
) -> Tuple[Union[FunctionProto, GraphProto, ModelProto], List[NodeProto]]:
    """
    Inlines functions in an ONNX graph.

    :param obj: onnx graph, FunctionProto, GraphProto,
        ModelProto
    :param protos: if None, the function assumes *obj* is of type
        ModelProto and the goal is to inline every function.
        If *protos* a list of strings, the function only inlines the
        functions in that list. If *protos* is a dictionary
        `{ (domain, type): FunctionProto }`, the function replaces every
        node `(domain, type)` by the code given in this dictionary
    :param existing_names: no new name will be taken in that set
    :param verbose: verbosity
    :return: modified object, list of modified nodes
    """
    if isinstance(obj, ModelProto):
        if verbose > 0:
            print("[onnx_inline_function] type=%r graph=%d" % (type(obj), id(obj)))
        if protos is None:
            fct = [f.name for f in obj.functions]
            ex_names = set(enumerate_onnx_names(obj))
            if existing_names is not None:
                ex_names |= existing_names
            return onnx_inline_function(
                obj, fct, existing_names=ex_names, verbose=verbose
            )
        if isinstance(protos, list):
            ex_names = set(enumerate_onnx_names(obj))
            if existing_names is not None:
                ex_names |= existing_names
            protos = {(f.domain, f.name): f for f in obj.functions}
            return onnx_inline_function(
                obj, protos, existing_names=ex_names, verbose=verbose
            )
    if isinstance(protos, list):
        protos = {(f.domain, f.name): f for f in protos}
    assert isinstance(
        protos, dict
    ), "obj is of type %r and protos must be a dictionary not %r." % (
        type(obj),
        type(protos),
    )

    if isinstance(obj, ModelProto):
        new_graph, m = onnx_inline_function(obj.graph, protos, verbose=verbose)
        if len(new_graph.initializer) != len(obj.graph.initializer):
            raise RuntimeError(
                "Mismatched number of initializers %d != %d."
                % (len(new_graph.initializer), len(obj.graph.initializer))
            )
        if len(new_graph.sparse_initializer) != len(obj.graph.sparse_initializer):
            raise RuntimeError(
                "Mismatched number of initializers %d != %d."
                % (len(new_graph.sparse_initializer), len(obj.graph.sparse_initializer))
            )
        new_functions = []
        distri = Counter((n.domain, n.op_type) for n in enumerate_onnx_nodes(new_graph))
        opsets = {op.domain: op.version for op in obj.opset_import}
        for f in obj.functions:
            key = f.domain, f.name
            if key not in protos:
                new_functions.append(f)
            elif key in distri:
                raise RuntimeError(
                    "Function %r still appears in the graph, "
                    "distibution=%s." % (key, pprint.pformat(distri))
                )
            if f.domain not in opsets:
                opsets[f.domain] = 1
        return (
            make_model(
                new_graph,
                functions=new_functions,
                opset_imports=[make_operatorsetid(k, v) for k, v in opsets.items()],
                producer_name=obj.producer_name,
                producer_version=obj.producer_version,
                ir_version=obj.ir_version,
                doc_string=obj.doc_string,
                domain=obj.domain,
                model_version=obj.model_version,
            ),
            m,
        )

    # FunctionProto, GraphProto
    if existing_names is None:
        existing_names = set(enumerate_onnx_names(obj))

    if verbose > 0:
        print("[onnx_inline_function] type=%r graph=%d begin" % (type(obj), id(obj)))
        distri = Counter((n.domain, n.op_type) for n in enumerate_onnx_nodes(obj))

    new_nodes = list(obj.node)
    modified_nodes = []
    n_iter = 0
    max_iter = onnx_subgraphs_level(obj) + 1
    modified = 1
    while modified > 0 and n_iter < max_iter:
        if verbose > 0:
            print(f"[onnx_inline_function] start iteration {n_iter!r}")

        # local context
        mapping = _inline_mapping(verbose, level=0)
        if isinstance(obj, GraphProto):
            mapping.update({i.name: i.name for i in obj.initializer})
            mapping.update({i.name: i.name for i in obj.sparse_initializer})
            for i in obj.input:
                if i.name not in mapping:
                    mapping[i.name] = i.name
        elif isinstance(obj, FunctionProto):
            mapping.update({i: i for i in obj.input})
        else:
            raise TypeError(f"Unexpected type for obj: {type(obj)!r}.")

        # loop on nodes
        old_nodes = new_nodes
        modified = 0
        new_nodes = []
        for node in old_nodes:
            nnodes, m = _onnx_inline_function_node(
                node, protos, existing_names, verbose, level=0
            )
            mapping.update({o: o for o in node.output})

            if len(m) > 0:
                if verbose > 0:
                    print(
                        "[onnx_inline_function] replaced node %r (%r) "
                        "with %d nodes (id=%r) -- %r -> %r (iter=%r)"
                        % (
                            node.name,
                            node.op_type,
                            len(nnodes),
                            id(node),
                            node.input,
                            node.output,
                            n_iter,
                        )
                    )
                modified += len(m)
                new_nodes.extend(nnodes)
                modified_nodes.extend(m)
            else:
                has_graph = False
                new_attributes = []
                for att in node.attribute:
                    if (
                        att.type == AttributeProto.GRAPH
                        and hasattr(att, "g")
                        and att.g is not None
                    ):
                        g, m = _onnx_inline_function_graph(
                            att.g,
                            protos,
                            verbose=verbose,
                            existing_names=existing_names,
                            mapping=mapping,
                            rename=False,
                            level=1,
                        )
                        if len(m) > 0:
                            modified_nodes.extend(m)
                            modified_nodes.append(node)
                            modified += 1 + len(m)
                            has_graph = True
                            att = make_attribute(att.name, g)
                    new_attributes.append(att)
                if has_graph:
                    new_node = make_node(
                        node.op_type,
                        node.input,
                        node.output,
                        domain=node.domain,
                        name=node.name,
                    )
                    new_node.attribute.extend(new_attributes)
                    new_nodes.append(new_node)
                else:
                    # we still need to check that this subgraph does
                    # not include a function
                    new_nodes.append(node)

        n_iter += 1
        if verbose > 0:
            total_node = len(list(enumerate_onnx_nodes(new_nodes)))
            print(
                "[onnx_inline_function] n_iter=%r/%r nodes=%r modified=%r "
                "n_nodes=%d total=%d"
                % (
                    n_iter,
                    max_iter,
                    len(obj.node),
                    modified,
                    len(new_nodes),
                    total_node,
                )
            )

    if verbose > 0:
        print(
            "[onnx_inline_function] type=%r graph=%d end with %d "
            "modified nodes" % (type(obj), id(obj), len(modified_nodes))
        )
        distri2 = Counter(
            (n.domain, n.op_type) for n in enumerate_onnx_nodes(new_nodes)
        )
        if distri != distri2:
            print("[onnx_inline_function] BEFORE")
            for k, v in sorted(distri.items()):
                print("[onnx_inline_function] %d -- %s" % (v, k))
            print("[onnx_inline_function] AFTER")
            for k, v in sorted(distri2.items()):
                print("[onnx_inline_function] %d -- %s" % (v, k))

    if isinstance(obj, FunctionProto):
        return (
            make_function(
                domain=obj.domain,
                fname=obj.name,
                inputs=obj.input,
                outputs=obj.output,
                nodes=new_nodes,
                opset_imports=[
                    make_operatorsetid(op.domain, op.version) for op in obj.opset_import
                ],
                doc_string=obj.doc_string,
                attributes=obj.attribute,
            ),
            modified_nodes,
        )
    if isinstance(obj, GraphProto):
        return (
            make_graph(
                new_nodes,
                obj.name,
                list(obj.input),
                list(obj.output),
                list(obj.initializer),
                doc_string=obj.doc_string,
                sparse_initializer=list(obj.sparse_initializer),
            ),
            modified_nodes,
        )
    raise TypeError(f"Unexpected type for obj {type(obj)!r}.")
