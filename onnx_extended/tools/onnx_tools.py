import os
from typing import Dict, Generator, Iterator, Optional, Set, Tuple, Union
import onnx

_rev_type: Dict[int, str] = {
    getattr(onnx.TensorProto, k): k
    for k in dir(onnx.TensorProto)
    if isinstance(getattr(onnx.TensorProto, k), int)
}


def load_model(
    model: Union[str, onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    external: bool = True,
    base_dir: Optional[str] = None,
) -> Union[onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto]:
    """
    Loads a model or returns the only argument if the type
    is already a ModelProto.

    :param model: proto file
    :param external: loads the external data as well
    :param base_dir: needed if external is True and
        the model has external weights
    :return: ModelProto
    """
    if isinstance(model, onnx.ModelProto):
        if base_dir is not None and external:
            if not os.path.exists(base_dir):
                raise FileNotFoundError(f"Unable to find folder {base_dir!r}.")
            onnx.load_external_data_for_model(model, base_dir)
        return model
    if isinstance(model, (onnx.GraphProto, onnx.FunctionProto)):
        return model
    if not os.path.exists(model):
        raise FileNotFoundError(f"Unable to find model {model!r}.")
    with open(model, "rb") as f:
        return onnx.load(f, load_external_data=external)


def load_external(
    model: onnx.ModelProto, base_dir: str, names: Optional[Set[str]] = None
):
    """
    Loads external data into memory.

    :param model: the model loaded with :func:`load_model`
    :param base_dir: directory when the data can be found
    :param names: subsets of names to load or None for all
    """
    from onnx.external_data_helper import (
        _get_all_tensors,
        uses_external_data,
        load_external_data_for_tensor,
    )

    for tensor in _get_all_tensors(model):
        if names is not None and tensor.name not in names:
            continue
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
            # After loading raw_data from external_data, change the state of tensors
            tensor.data_location = onnx.TensorProto.DEFAULT
            # and remove external data
            del tensor.external_data[:]


def enumerate_model_tensors(
    model: onnx.ModelProto,
) -> Iterator[Tuple[onnx.TensorProto, bool]]:
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


def save_model(
    proto: onnx.ModelProto,
    filename: str,
    external: bool = False,
    convert_attribute: bool = True,
    size_threshold: int = 1024,
    all_tensors_to_one_file: bool = True,
):
    """
    Saves a model into an onnx file.

    :param proto: ModelProto
    :param filename: where to save it
    :param external: saves weights as external data
    :param convert_attribute: converts attributes as well
    :param size_threshold: every weight above that threshold is saved as external
    :param all_tensors_to_one_file: saves all tensors in one unique file
    """
    if not external:
        onnx.save_model(proto, filename)
        return

    dirname, shortname = os.path.split(filename)
    onnx.convert_model_to_external_data(
        proto,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=shortname + ".data",
        convert_attribute=convert_attribute,
        size_threshold=size_threshold,
    )

    proto = onnx.write_external_data_tensors(proto, dirname)
    with open(filename, "wb") as f:
        f.write(proto.SerializeToString())


def _info_type(
    typ: Union[onnx.TensorProto, onnx.TypeProto, onnx.SparseTensorProto]
) -> Dict[str, str]:
    if typ is None:
        return {}
    if isinstance(typ, (onnx.TensorProto, onnx.SparseTensorProto)):
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
    model: Union[str, onnx.ModelProto, onnx.GraphProto],
    level: int = 0,
    shapes: Optional[Dict[str, onnx.TypeProto]] = None,
    external: bool = True,
) -> Generator[Dict[str, Union[str, float]], None, None]:
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
    if shapes is None and isinstance(proto, onnx.ModelProto):
        p2 = onnx.shape_inference.infer_shapes(proto)
        values = p2.graph.value_info
        shapes = {}
        for value in values:
            shapes[value.name] = value.type
        for o in proto.graph.output:
            if o.name not in shapes:
                shapes[o.name] = o.type

    if isinstance(proto, onnx.ModelProto):
        if shapes is None:
            raise RuntimeError("shape inference has failed.")
        for item in enumerate_onnx_node_types(proto.graph, level=level, shapes=shapes):
            yield item

    elif isinstance(model, onnx.FunctionProto):
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
                if att.type == onnx.AttributeProto.GRAPH:
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
