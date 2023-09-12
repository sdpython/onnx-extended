import os
import re
import io
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import (
    FunctionProto,
    GraphProto,
    ModelProto,
    TensorProto,
    SequenceProto,
    ValueInfoProto,
    load,
)
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.numpy_helper import to_array
from onnx.reference import ReferenceEvaluator


def _type_shape(
    input_def: Union[str, ValueInfoProto]
) -> Tuple[Any, Tuple[int, ...], str]:
    if isinstance(input_def, str):
        reg = re.compile(
            "([a-z][a-z0-9]*)?([(]([ a-zA-Z,0-9]+)[)])?(:([A-Z][A-Z0-9]*))?"
        )
        search = reg.match(input_def)
        if search is None:
            raise ValueError(f"Unable to interpret string {input_def!r}.")
        grs = search.groups()
        dtype = grs[0]
        shape = None if grs[2] is None else grs[2].replace(" ", "").split(",")
        law = grs[-1]
        new_shape = []
        if shape is not None:
            for i in shape:
                try:
                    vi = int(i)
                    new_shape.append(vi)
                except ValueError:
                    new_shape.append(i)
            rshape = tuple(new_shape)
        else:
            rshape = None
        dt = None if dtype is None else getattr(np, dtype)
        return dt, rshape, law

    if isinstance(input_def, ValueInfoProto):
        try:
            ttype = input_def.type.tensor_type
        except AttributeError:
            raise ValueError(f"Unsupported input type {input_def!r}.")
        dt = ttype.elem_type
        new_shape = []
        for d in ttype.shape.dim:
            if d.dim_param:
                new_shape.append(d.dim_param)
            else:
                new_shape.append(d.dim_value)
        ndt = tensor_dtype_to_np_dtype(dt)
        return ndt, tuple(new_shape)

    raise TypeError(f"Unexpected type {type(input_def)} for input_def.")


def _generate_random_inputs(
    dtype: Any,
    shape: Tuple[Union[int, str], ...],
    law: Optional[str] = None,
    dims: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Creates random or specific inputs.

    :param dtype: numpy dtype
    :param shape: expected shape
    :param law: law of the coefficients, default is 'U10', uniform law
    :param dims: letter are allowed, contains the named dimensions already
        mapped to a specific value
    :return: tuple (array, updated dims)

    Dimensions are modified inplace.
    """
    if dims is None:
        dims = {}
    if law is None:
        law = "U10"
    new_shape = []
    for sh in shape:
        if isinstance(sh, int):
            new_shape.append(sh)
        elif isinstance(sh, str):
            if sh not in dims:
                dims[sh] = 8
            new_shape.append(dims[sh])
    final_shape = tuple(new_shape)

    if law == "U10":
        res = np.random.random(final_shape).astype(dtype)
        return res, dims

    raise ValueError(f"Unexpected value for law={law!r}.")


def store_intermediate_results(
    model: Union[ModelProto, str],
    inputs: List[Union[str, np.ndarray, TensorProto]],
    out: str = ".",
    runtime: Union[type, str] = "CReferenceEvaluator",
    providers: Union[str, List[str]] = "CPU",
    verbose: int = 0,
):
    """
    Executes an onnx model with a runtime and stores the
    intermediate results in a folder.
    See :class:`CReferenceEvaluator <onnx_extended.reference.CReferenceEvaluator>`
    for further details.

    :param model: path to a model or ModelProto
    :param inputs: list of inputs for the model
    :param out: output path
    :param runtime: runtime class to use
    :param providers: list of providers
    :param verbose: verbosity level
    :return: outputs
    """
    if isinstance(model, str):
        if not os.path.exists(model):
            raise FileNotFoundError(f"File {model!r} does not exists.")
    if isinstance(providers, str):
        providers = [s.strip() for s in providers.split(",")]
    if runtime == "CReferenceEvaluator":
        from .reference import CReferenceEvaluator

        cls_runtime = CReferenceEvaluator
        add_providers = False
    elif hasattr(runtime, "run"):
        cls_runtime = runtime
        add_providers = True
    else:
        raise ValueError(f"Unexpected runtime {runtime!r}.")
    iv = int(verbose)
    if add_providers:
        inst = cls_runtime(
            model, providers=providers, save_intermediate=out, verbose=iv
        )
    else:
        inst = cls_runtime(model, save_intermediate=out, verbose=iv)
    names = inst.input_names
    if len(names) < len(inputs):
        raise RuntimeError(
            f"There are more inputs ({len(inputs)}) "
            f"than names ({len(names)}). Names are {names}."
        )

    dims = {}
    feeds = {}
    for i, (name, inp) in enumerate(zip(names, inputs)):
        if isinstance(inp, str) and os.path.exists(inp):
            with open(inp, "rb") as f:
                vect = f.read()
            tp = TensorProto()
            tp.ParseFromString(vect)
            value = to_array(tp)
        else:
            ty, shape, law = _type_shape(inp)
            if ty is None or shape is None:
                if isinstance(inst, ReferenceEvaluator):
                    ty, shape = _type_shape(inst.proto_.graph.input[i])
                else:
                    raise RuntimeError(
                        f"shape or dtype is unknown and cannot "
                        f"be retrieved from class {type(inst)}."
                    )
            value, dims = _generate_random_inputs(ty, shape, law, dims=dims)
        feeds[name] = value

    got = inst.run(None, feeds)
    return got


def display_intermediate_results(
    model: str, save: Optional[str] = None, tab: int = 12, fprint: Callable = print
):
    """
    Displays shape, type for a model.

    :param model: a model
    :param save: save the results as a dataframe
    :param tab: column size for the output
    :param fprint: function to print
    """
    from .tools.onnx_tools import enumerate_onnx_node_types

    if save is not None:
        ext = os.path.splitext(save)[-1]
        if ext not in {".csv", ".xlsx"}:
            raise ValueError(f"Unexpected format {save!r}, extension is {ext!r}.")
    else:
        exts = None

    def _fixed(s, length=10):
        if not isinstance(s, str):
            raise TypeError(f"Unexpected type {type(s)}: {s!r}.")
        return (
            (s[: length - 1] + " ")
            if len(s) >= length - 1
            else s + " " * (length - len(s))
        )

    n_rows = 0
    rows = []
    for obs in enumerate_onnx_node_types(model):
        if "level" not in obs:
            raise RuntimeError(f"Unexpected value obs={obs!r}.")
        indent = " " * obs["level"] * tab
        values = [
            indent,
            _fixed(obs.get("kind", ""), tab),
            _fixed(obs.get("type", ""), tab),
            _fixed(obs.get("name", ""), tab),
            _fixed(obs.get("elem_type", ""), tab),
            _fixed(obs.get("shape", ""), tab),
            _fixed(obs.get("input_types", ""), tab),
            _fixed(obs.get("output_types", ""), tab),
            _fixed(obs.get("inputs", ""), tab),
            _fixed(obs.get("outputs", ""), tab),
        ]
        line = "".join(values)
        fprint(line)

        n_rows += 1
        if save is not None:
            rows.append(obs)

    if n_rows == 0:
        if isinstance(model, str):
            raise RuntimeError(f"Model {model!r} is empty.")
        raise RuntimeError(f"Model type {type(model)} is empty.")

    if save is not None:
        from pandas import DataFrame

        df = DataFrame(rows)
        exts = {".csv": df.to_csv, ".xlsx": df.to_excel}
        exts[ext](save, index=False)


def print_proto(proto: str, fmt: str = "raw"):
    """
    Shows an onnx model or a protobuf string on stdout.
    Extension '.onnx' is considered a model,
    extension '.proto' or '.pb' is a protobuf string.

    :param proto: a file
    :param fmt: format to use to print the model,
        `raw` prints out the string produced by `print(model)`,
        `nodes` only prints out the node name
    """
    if isinstance(proto, str):
        if not os.path.exists(proto):
            raise FileNotFoundError(f"Unable to find file {proto!r}.")
        ext = os.path.splitext(proto)[-1]
        if ext == ".onnx":
            with open(proto, "rb") as f:
                proto_loaded = load(f)
        elif ext in (".pb", ".proto"):
            with open(proto, "rb") as f:
                content = f.read()
            exc = []
            proto_loaded = None
            for cls in [
                TensorProto,
                SequenceProto,
                FunctionProto,
                ModelProto,
                GraphProto,
            ]:
                inst = cls()
                try:
                    inst.ParseFromString(content)
                    proto_loaded = inst
                    break
                except Exception as e:
                    exc.append((cls, e))
            if proto_loaded is None:
                msg = "\n".join(f"type: {c}: {e}" for c, e in exc)
                raise RuntimeError(f"Unable to load {proto!r}, tried:\n{msg}")
        else:
            raise ValueError(f"Unexpected file extension {ext!r} for file {proto!r}.")
    else:
        proto_loaded = proto

    print(f"Type: {type(proto_loaded)}")
    if fmt == "raw":
        print(proto_loaded)
    elif fmt == "nodes":
        from .tools.graph.onnx_graph_struct import Graph

        graph = Graph(proto_loaded)
        for node in graph:
            print(str(node).replace("<parent>, ", ""))
    else:
        raise ValueError(f"Unexpected value for fmt={fmt!r}.")


def cmd_quantize(
    model: Union[ModelProto, str],
    output: Optional[str] = None,
    kind: str = "fp8",
    scenario: str = "onnxruntime",
    early_stop: Optional[int] = None,
    quiet: bool = False,
    verbose: int = 0,
    index_transpose: int = 2,
    exceptions: Optional[List[Dict[str, str]]] = None,
):
    """
    Quantizes a model

    :param model: path to a model or ModelProto
    :param output: output file
    :param kind: kind of quantization
    :param scenario: depends on the quantization
    :param early_stop: stops early to see the preliminary results
    :param quiet: do not stop an exception
    :param verbose: verbosity level
    :param index_transpose: which input to transpose before calling gemm:
        0 (none), 1 (first), 2 (second), 3 for both
    :param exceptions: exclude nodes from the quantization,
        `[{"name": "node_name1"}, {"name": "node_name2"}]` will exclude
        these two node names from the quantization
    """
    from .tools.graph import Graph

    if isinstance(model, str):
        if not os.path.exists(model):
            raise FileNotFoundError(f"Unable to find file {model!r}.")
        ext = os.path.splitext(model)[-1]
        if ext == ".onnx":
            with open(model, "rb") as f:
                proto_loaded = load(f)
    else:
        proto_loaded = model
    graph = Graph(proto_loaded)

    if verbose:
        logging.basicConfig(
            level=logging.WARN
            if verbose > 2
            else (logging.DEBUG if verbose > 1 else logging.INFO)
        )

    if kind == "fp8":
        from .tools.graph import quantize_float8

        logger = logging.getLogger("onnx-extended")
        logger.info("Model initial size: %d", len(proto_loaded.SerializeToString()))
        new_graph = quantize_float8(
            graph,
            early_stop=early_stop or -1,
            quiet=quiet,
            version=scenario,
            index_transpose=index_transpose,
            exceptions=exceptions,
        )
        if new_graph is None:
            logger.warning("No node was quantized.")
            return
        onx2 = new_graph.to_onnx()
        seq = onx2.SerializeToString()
        logger.info("Model quantized size: %d", len(seq))
        with open(output, "wb") as f:
            f.write(seq)
        return

    if kind == "fp16":
        from .tools.graph import cast_constant

        logger = logging.getLogger("onnx-extended")
        logger.info("Model initial size: %d", len(proto_loaded.SerializeToString()))
        new_graph = cast_constant(
            graph,
            quiet=quiet,
            from_type=TensorProto.FLOAT,
            to_type=TensorProto.FLOAT16,
        )
        if new_graph is None:
            logger.warning("No node was modified.")
            return
        onx2 = new_graph.to_onnx()
        seq = onx2.SerializeToString()
        logger.info("Model reduced size: %d", len(seq))
        with open(output, "wb") as f:
            f.write(seq)
        return

    raise ValueError(f"Unexpected value {kind!r} for kind.")


def cmd_select(
    model: Union[ModelProto, str],
    save: Optional[str] = None,
    inputs: Optional[Union[str, List[str]]] = None,
    outputs: Optional[Union[str, List[str]]] = None,
    verbose: int = 0,
):
    """
    Selects a subgraph in a model.

    :param model: path to a model or ModelProto
    :param save: model ot save in this file
    :param inputs: list of inputs or empty to keep the original inputs
    :param outputs: list of outputs or empty to keep the original outputs
    :param verbose: verbosity level
    """
    from .tools.onnx_manipulations import select_model_inputs_outputs

    if isinstance(model, str):
        if not os.path.exists(model):
            raise FileNotFoundError(f"Unable to find file {model!r}.")
        ext = os.path.splitext(model)[-1]
        if ext == ".onnx":
            with open(model, "rb") as f:
                proto_loaded = load(f)
    else:
        proto_loaded = model

    if verbose:
        logging.basicConfig(
            level=logging.WARN
            if verbose > 2
            else (logging.DEBUG if verbose > 1 else logging.INFO)
        )

    if isinstance(inputs, str):
        inputs = inputs.strip().split(",")
    if isinstance(outputs, str):
        outputs = outputs.strip().split(",")

    logger = logging.getLogger("onnx-extended")
    logger.info("Initial model size: %d", len(proto_loaded.SerializeToString()))
    onx2 = select_model_inputs_outputs(
        proto_loaded,
        inputs=inputs,
        outputs=outputs,
        verbose=verbose,
    )
    seq = onx2.SerializeToString()
    logger.info("Selected model size: %d", len(seq))
    with open(save, "wb") as f:
        f.write(seq)


def plot_profile(
    filename: str,
    kind: str,
    out_csv: Optional[str] = None,
    out_png: Optional[str] = None,
    title: Optional[str] = None,
    with_shape: bool = False,
    verbose: int = 0,
):
    """
    Plots a profiling.

    :param filename: raw data to load
    :param kind: kind of plot to so, see below
    :param out_csv: output the data into that csv file
    :param out_png: output the graph in that file
    :param with_shape: consider input shape when showing results
    :param title: title (optional)
    :param verbose: verbosity, if > 0, prints out the data in csv format
    """
    import matplotlib.pyplot as plt
    from .tools.js_profile import (
        js_profile_to_dataframe,
        plot_ort_profile,
        _preprocess_graph1,
        _preprocess_graph2,
    )

    if verbose:
        print(f"[plot_profile] load {filename!r}")

    if kind == "profile_op":
        df = js_profile_to_dataframe(filename, first_it_out=True, with_shape=with_shape)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax, title=title)
        df = _preprocess_graph1(df)
    elif kind == "profile_node":
        df = js_profile_to_dataframe(
            filename, first_it_out=True, agg=True, with_shape=with_shape
        )

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plot_ort_profile(df, ax, title=title)
        df = _preprocess_graph2(df)
    else:
        raise ValueError(f"Unexpected kind {kind:r}.")

    if verbose:
        st = io.StringIO()
        df.to_csv(st)
        print(st.getvalue())

    if out_csv not in {"", None}:
        if verbose:
            print(f"[plot_profile] save {out_csv!r}")
        df.to_csv(out_csv)

    if out_png not in {"", None}:
        if verbose:
            print(f"[plot_profile] save {out_png!r}")
        fig.savefig(out_png)


def cmd_plot(
    filename: str,
    kind: str,
    out_csv: Optional[str] = None,
    out_png: Optional[str] = None,
    title: Optional[str] = None,
    with_shape: bool = False,
    verbose: int = 0,
):
    """
    Plots a graph.

    :param filename: raw data to load
    :param kind: kind of plot to so, see below
    :param out_csv: output the data into that csv file
    :param out_png: output the graph in that file
    :param title: title (optional)
    :param with_shape: keep the shape to aggregate
    :param verbose: verbosity, if > 0, prints out the data in csv format

    Kinds of plots:

    * `'profile_op'`: draws the profiling per node type
    * `'profile_node'`: draws the profiling per node
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Unable to find {filename!r}.")
    allowed = {"profile_op", "profile_node"}
    if kind in allowed:
        plot_profile(filename, kind, out_csv, out_png, title=title, verbose=verbose)
    else:
        raise ValueError(f"Unexpected kind {kind:r}, it should be {allowed}.")
