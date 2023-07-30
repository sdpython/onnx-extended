import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, TensorProto, ValueInfoProto
from onnx.helper import tensor_dtype_to_np_dtype


def _type_shape(input_def: Union[str, ValueInfoProto]) -> Tuple[Any, Tuple[int, ...]]:
    if isinstance(input_def, str):
        reg = re.compile("([a-z][a-z0-9]*)[(]([ a-zA-Z,0-9]+)[)]")
        search = reg.match(input_def)
        if search is None:
            raise ValueError(f"Unable to interpret string {input_def!r}.")
        dtype, shape = search.groups()
        shape = shape.replace(" ", "").split(",")
        new_shape = []
        for i in shape:
            try:
                vi = int(i)
                new_shape.append(vi)
            except ValueError:
                new_shape.append(i)
        dt = getattr(np, dtype)
        return dt, tuple(new_shape)

    if isinstance(input_def, ValueInfoProto):
        try:
            ttype = input_def.type.tensor_type
        except AttributeError:
            raise ValueError(f"Unsupported input type {input_def!r}.")
        dt = ttype.elem_ttype
        new_shape = []
        for d in ttype.shape.dims:
            if d.dim_param:
                new_shape.append(d.dim_param)
            else:
                new_shape.append(d.dim_value)
        ndt = tensor_dtype_to_np_dtype(dt)
        return ndt, tuple(new_shape)

    raise TypeError(f"Unexpected type {type(input_def)} for input_def.")


def create_random_input(
    input_def: Union[str, ValueInfoProto], dims: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Creates random or specific inputs.

    :param input_def: a string `float32(4,5)` or `float32(N,5)` or a instance of
        type :class:`onnx.ValueInfoProto`
    :param dims: letter are allowed, contains the named dimensions already
        mapped to a specific value
    :return: tuple (array, updated dims)

    Dimension are modified inplace.
    """
    if dims is None:
        dims = {}
    typ, shape = _type_shape(input_def)


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

    :param model: path to a model of ModelProto
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
    elif hasattr(runtime, "run"):
        cls_runtime = runtime
    else:
        raise ValueError(f"Unexpected runtime {runtime!r}.")
    inst = cls_runtime(
        model, providers=providers, save_intermediate=out, verbose=int(verbose)
    )
    got = inst.run(inputs)
    return got
