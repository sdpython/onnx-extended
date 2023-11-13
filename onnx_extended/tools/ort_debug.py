from typing import Any, Dict, Iterator, List, Optional, Union, Tuple
import numpy as np
from onnx import AttributeProto, ModelProto, NodeProto, load
from onnx.reference.op_run import to_array_extended
from .onnx_nodes import select_model_inputs_outputs


def render_node(node: NodeProto) -> str:
    """
    Renders a node into text to display it.

    :param node: Node
    :return: trext
    """
    att_display = [
        "activations",
        "align_corners",
        "allowzero",
        "alpha",
        "auto_pad",
        "axis",
        "axes",
        "batch_axis",
        "batch_dims",
        "beta",
        "bias",
        "blocksize",
        "case_change_action",
        "ceil_mode",
        "center_point_box",
        "clip",
        "coordinate_transformation_mode",
        "count_include_pad",
        "cubic_coeff_a",
        "decay_factor",
        "detect_negative",
        "detect_positive",
        "dilation",
        "dilations",
        "direction",
        "dtype",
        "end",
        "epsilon",
        "equation",
        "exclusive",
        "exclude_outside",
        "extrapolation_value",
        "fmod",
        "gamma",
        "group",
        "hidden_size",
        "high",
        "ignore_index",
        "input_forget",
        "is_case_sensitive",
        "k",
        "keepdims",
        "kernel_shape",
        "lambd",
        "largest",
        "layout",
        "linear_before_reset",
        "locale",
        "low",
        "max_gram_length",
        "max_skip_count",
        "mean",
        "min_gram_length",
        "mode",
        "momentum",
        "nearest_mode",
        "ngram_counts",
        "ngram_indexes",
        "noop_with_empty_axes",
        "norm_coefficient",
        "norm_coefficient_post",
        "num_scan_inputs",
        "output_height",
        "output_padding",
        "output_shape",
        "output_width",
        "p",
        "padding_mode",
        "pads",
        "perm",
        "pooled_shape",
        "reduction",
        "reverse",
        "sample_size",
        "sampling_ratio",
        "scale",
        "scan_input_axes",
        "scan_input_directions",
        "scan_output_axes",
        "scan_output_directions",
        "seed",
        "select_last_index",
        "size",
        "sorted",
        "spatial_scale",
        "start",
        "storage_order",
        "strides",
        "time_axis",
        "to",
        "training_mode",
        "transA",
        "transB",
        "type",
        "upper",
        "xs",
        "y",
        "zs",
    ]

    sub_graphs_names: Dict[str, str] = {}

    def _get_subgraph_name(idg):
        if idg in sub_graphs_names:
            return sub_graphs_names[idg]
        g = "G%d" % (len(sub_graphs_names) + 1)
        sub_graphs_names[idg] = g
        return g

    def str_node(indent, node):
        atts = []
        if hasattr(node, "attribute"):
            for att in node.attribute:
                done = True
                if hasattr(att, "ref_attr_name") and att.ref_attr_name:
                    atts.append(f"{att.name}=${att.ref_attr_name}")
                    continue
                if att.name in att_display:
                    if att.type == AttributeProto.INT:
                        atts.append("%s=%d" % (att.name, att.i))
                    elif att.type == AttributeProto.FLOAT:
                        atts.append(f"{att.name}={att.f:1.2f}")
                    elif att.type == AttributeProto.INTS:
                        atts.append(
                            "%s=%s" % (att.name, str(list(att.ints)).replace(" ", ""))
                        )
                    else:
                        done = False
                elif (
                    att.type == AttributeProto.GRAPH
                    and hasattr(att, "g")
                    and att.g is not None
                ):
                    atts.append(f"{att.name}={_get_subgraph_name(id(att.g))}")
                else:
                    done = False
                if done:
                    continue
                if att.type in (
                    AttributeProto.TENSOR,
                    AttributeProto.TENSORS,
                    AttributeProto.SPARSE_TENSOR,
                    AttributeProto.SPARSE_TENSORS,
                ):
                    try:
                        val = str(to_array_extended(att.t).tolist())
                    except TypeError as e:
                        raise TypeError(
                            "Unable to display tensor type %r.\n%s"
                            % (att.type, str(att))
                        ) from e
                    if "\n" in val:
                        val = val.split("\n", maxsplit=1) + "..."
                    if len(val) > 10:
                        val = val[:10] + "..."
                elif att.type == AttributeProto.STRING:
                    val = str(att.s)
                    if len(val) > 50:
                        val = val[:40] + "..." + val[-10:]
                elif att.type == AttributeProto.STRINGS:
                    n_val = list(att.strings)
                    if len(n_val) < 5:
                        val = ",".join(map(str, n_val))
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:2])),
                            ",".join(map(str, n_val[-2:])),
                        )
                elif att.type == AttributeProto.INT:
                    val = str(att.i)
                elif att.type == AttributeProto.FLOAT:
                    val = str(att.f)
                elif att.type == AttributeProto.INTS:
                    n_val = list(att.ints)
                    if len(n_val) < 6:
                        val = f"[{','.join(map(str, n_val))}]"
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:3])),
                            ",".join(map(str, n_val[-3:])),
                        )
                elif att.type == AttributeProto.FLOATS:
                    n_val = list(att.floats)
                    if len(n_val) < 5:
                        val = f"[{','.join(map(str, n_val))}]"
                    else:
                        val = "%d:[%s...%s]" % (
                            len(n_val),
                            ",".join(map(str, n_val[:2])),
                            ",".join(map(str, n_val[-2:])),
                        )
                else:
                    val = ".%d" % att.type
                atts.append(f"{att.name}={val}")
        inputs = list(node.input)
        if atts:
            inputs.extend(atts)
        domain = "" if node.domain in ("", "ai.onnx.ml") else f"[{node.domain}]"
        return "%s%s%s(%s) -> %s" % (
            "  " * indent,
            node.op_type,
            domain,
            ", ".join(inputs),
            ", ".join(node.output),
        )

    return str_node(0, node)


def enumerate_ort_run(
    onx: Union[str, ModelProto],
    feeds: Dict[str, Any],
    verbose: int = 0,
    providers: Optional[List[str]] = None,
    **kwargs: Dict[str, Any],
) -> Iterator[Tuple[List[str], List[Any], NodeProto]]:
    """
    Yields all the intermediate results produced by
    :epkg:`onnxruntime`.

    :param onx: model
    :param feeds: input tensors
    :param verbose: prints out a summary of the results
    :param providers: if not specified, default is `["CPUExecutionProvider"]`
    :param kwargs: additional parameter to give InferenceSession
        when it is initialized
    :return: intermediate results, names, and node
    """
    from onnxruntime import InferenceSession

    if providers is None:
        providers = ["CPUExecutionProvider"]
    if isinstance(onx, str):
        with open(onx, "rb") as f:
            proto = load(f)
    else:
        proto = onx

    inputs = [i.name for i in proto.graph.input]
    if verbose == 1:
        import tqdm

        loop = tqdm.tqdm(proto.graph.node)
    else:
        loop = proto.graph.node
    if verbose > 1:
        for init in proto.graph.initializer:
            value = to_array_extended(init)
            if verbose <= 2:
                print(" +C %s: %s%s" % (init.name, value.dtype, value.shape))
            elif value.size < 10:
                print(
                    " +C %s: %s%s = %s"
                    % (
                        init.name,
                        value.dtype,
                        value.shape,
                        str(value).replace("\n", ""),
                    )
                )
            else:
                print(
                    " +C %s: %s%s ~ %s..."
                    % (
                        init.name,
                        value.dtype,
                        value.shape,
                        str(value.ravel()[:8]).replace("\n", ""),
                    )
                )
        for i in onx.graph.input:
            if i.name not in feeds:
                continue
            value = feeds[i.name]
            if verbose <= 2:
                print(" +I %s: %s%s" % (i.name, value.dtype, value.shape))
            elif value.size < 10:
                print(
                    " +I %s: %s%s = %s"
                    % (
                        i.name,
                        value.dtype,
                        value.shape,
                        str(value).replace("\n", ""),
                    )
                )
            else:
                print(
                    " +I %s: %s%s ~ %s..."
                    % (
                        i.name,
                        value.dtype,
                        value.shape,
                        str(value.ravel()[:8]).replace("\n", ""),
                    )
                )

    for node in loop:
        names = list(node.output)
        if verbose > 1:
            print(render_node(node))
        subproto = select_model_inputs_outputs(proto, outputs=names, inputs=inputs)
        sess = InferenceSession(
            subproto.SerializeToString(), providers=providers, **kwargs
        )
        outputs = sess.run(None, feeds)
        if verbose > 1:
            for name, value in zip(node.output, outputs):
                if isinstance(value, np.ndarray) and verbose <= 2:
                    print(" + %s: %s%s" % (name, value.dtype, value.shape))
                elif value.size < 10:
                    print(
                        " + %s: %s%s = %s"
                        % (name, value.dtype, value.shape, str(value).replace("\n", ""))
                    )
                else:
                    print(
                        " + %s: %s%s ~ %s..."
                        % (
                            name,
                            value.dtype,
                            value.shape,
                            str(value.ravel()[:8]).replace("\n", ""),
                        )
                    )
        yield names, outputs, node
