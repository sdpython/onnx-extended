import time
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union
import numpy
from onnx import AttributeProto, ModelProto, NodeProto, GraphProto, FunctionProto
from onnx.helper import make_model, make_node, make_graph, make_opsetid
from ...ext_test_case import measure_time


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
        f"Unable to find {name!r} among {[att.name for att in node.attribute]}."
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
    :param new_opset: new opset for the new domain, if not specified,
        it is 1 for any opset other than ""
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
        raise NotImplementedError("onx is FunctionProto, not implemented yet.")

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
    assert new_op_domain != op_domain or new_opset is None, (
        f"If new_op_domain=={new_op_domain!r}, "
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


def optimize_model(
    onx: ModelProto,
    feeds: Dict[str, numpy.ndarray],
    transform: Callable[[ModelProto], ModelProto],
    session: Callable[[ModelProto], Any],
    params: Dict[str, List[Any]],
    baseline: Optional[Callable[[ModelProto], Any]] = None,
    verbose: bool = False,
    number: int = 10,
    repeat: int = 10,
    warmup: int = 5,
    n_tries: int = 2,
    sleep: float = 0.1,
) -> List[Dict[str, Union[str, float]]]:
    """
    Optimizes a model by trying out many possibilities.

    :param onx: ModelProto
    :param feeds: inputs as a dictionary of numpy arrays
    :param transform: function taking a ModelProto and returning a ModelProto
        based on the values coming from *params*
    :param session: function which takes a modifed ModelProto
        and return a session
    :param params: dictionary of values to test `{ param_name: [ param_values ] }`
    :param baseline: function which takes a modifed ModelProto
        and return a session, identified as the baseline
    :param verbose: use :epkg:`tqdm` to show improvment
    :param number: parameter to :func:`measure_time
        <onnx_extended.ext_test_case.measure_time>`
    :param repeat: parameter to :func:`measure_time
        <onnx_extended.ext_test_case.measure_time>`
    :param warmup: parameter to :func:`measure_time
        <onnx_extended.ext_test_case.measure_time>`
    :param n_tries: number of times to measure, if the measurements returns
        very different results, values for *number* or *repeat* should
        be increased
    :param sleep: time to sleep between two measurements
    :return: list of results returned by :func:`measure_time
        <onnx_extended.ext_test_case.measure_time>`

    See example :ref:`l-plot-optim-tree-ensemble` for an example.
    """
    assert sleep < 1, f"sleep={sleep} >= 1, probably a mistake."
    keys = ["TRY", *list(params.keys())]
    sets = [list(range(n_tries))] + [params[k] for k in keys[1:]]
    loops = list(product(*sets))
    if verbose:
        from tqdm import tqdm

        loop = tqdm(loops)
    else:
        loop = loops

    res = []
    if baseline is not None:
        sess = baseline(onx)
        # one run to make run it is working
        sess.run(None, feeds)
        if sleep > 0:
            time.sleep(sleep)
        obs: Dict[str, Any] = measure_time(
            lambda sess=sess: sess.run(None, feeds),
            number=number,
            repeat=repeat,
            warmup=warmup,
        )
        obs["n_exp"] = 0
        obs["n_exp_name"] = "TRY=0,baseline"
        obs["short_name"] = "0,baseline"
        obs["TRY"] = 0
        obs["name"] = "baseline"
        res.append(obs)
        base_perf = obs["average"]
    else:
        base_perf = None

    min_perf = None
    _s = dict(
        parallel_tree="//tree",
        parallel_tree_N="//tree_N",
        parallel_N="//N",
        batch_size_tree="bs_tree",
        batch_size_ows="bs_rows",
        use_node3="n3",
    )

    for it, values in enumerate(loop):
        if verbose:
            msg = [f"i={it+1}/{len(loops)}"]
            msg.extend([f"{_s.get(k,k)}={v}" for k, v in zip(keys, values)])
            if min_perf and base_perf:
                msg.append(f" ~={base_perf/min_perf:1.2f}x")
            loop.set_description(" ".join(msg))

        kwargs = dict(zip(keys, values))
        del kwargs["TRY"]
        onx_modified = transform(onx, **kwargs)
        sess = session(onx_modified)
        if sleep > 0:
            time.sleep(sleep)
        obsl: Dict[str, Any] = measure_time(
            lambda sess=sess: sess.run(None, feeds),
            number=number,
            repeat=repeat,
            warmup=warmup,
        )
        if not min_perf or min_perf > obsl["average"]:
            min_perf = obsl["average"]
        obsl.update(kwargs)
        obsl["n_exp"] = it
        obsl["n_exp_name"] = ",".join(f"{k}={v}" for k, v in zip(keys, values))
        obsl["short_name"] = ",".join(f"{v}" for v in values)
        obsl["name"] = ",".join(f"{v}" for v in values[1:])
        res.append(obsl)

    if baseline is not None:
        for n in range(1, n_tries):
            sess = baseline(onx)
            if sleep > 0:
                time.sleep(sleep)
            obsf: Dict[str, Any] = measure_time(
                lambda sess=sess: sess.run(None, feeds),
                number=number,
                repeat=repeat,
                warmup=warmup,
            )
            obsf["n_exp"] = 0
            obsf["n_exp_name"] = f"TRY={n},baseline"
            obsf["short_name"] = f"{n},baseline"
            obsf["name"] = "baseline"
            obsf["TRY"] = n
            res.append(obsf)

    return res
