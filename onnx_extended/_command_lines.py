import os
from typing import List, Union
import numpy as np
from onnx import ModelProto, TensorProto


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
    inst.run(inputs)
    raise NotImplementedError("not finished with {inst!r}")
