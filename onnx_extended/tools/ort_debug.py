from typing import Any, Dict, Iterator, List, Union, Tuple
import numpy as np
from onnx import ModelProto, load
from onnxruntime import InferenceSession
from ..reference import from_array_extended
from .onnx_manipulations import select_model_inputs_outputs


def enumerate_ort_run(
    onx: Union[str, ModelProto], feeds: Dict[str, Any], verbose: int = 0
) -> Iterator[Tuple[List[str], List[Any]]]:
    """
    Yields all the intermediate results produced by
    :epkg:`onnxruntime`.

    :param onx: model
    :param feeds: input tensors
    :param verbose: prints out a summary of the results
    :return: intermediate results and names
    """
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
            value = from_array_extended(init)
            if verbose <= 2:
                print(" + %s: %s-%s" % (init.name, value.dtype, value.shape))
            else:
                print(" + %s: %s" % (init.name, value))
    for node in loop:
        names = list(node.output)
        subproto = select_model_inputs_outputs(proto, outputs=names, inputs=inputs)

        sess = InferenceSession(
            subproto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = sess.run(None, feeds)
        if verbose > 1:
            print(
                "%s(%s) -> %s"
                % (node.op_type, ", ".join(node.input), ", ".join(node.output))
            )
            for name, value in zip(node.output, outputs):
                if isinstance(value, np.ndarray) and verbose <= 2:
                    print(" + %s: %s%s" % (name, value.dtype, value.shape))
                else:
                    print(" + %s: %s%s" % (name, value.dtype, value.shape))
                    print(value)
        yield names, outputs
