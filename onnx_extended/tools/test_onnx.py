import os
from typing import List, Optional
import numpy as np
from onnx import ModelProto
from ..reference import CReferenceEvaluator, from_array_extended


def save_for_benchmark_or_test(
    folder: str,
    test_name: str,
    model: ModelProto,
    inputs: List[np.ndarray],
    outputs: Optional[List[np.ndarray]] = None,
    data_set: int = 0,
) -> str:
    """
    Saves input, outputs on disk to later uses it as a backend test
    or a benchmark.

    :param folder: folder to save
    :param test_name: test name or subfolder
    :param model: model to save
    :param inputs: inputs of the node
    :param outputs: outputs of the node, supposedly the expected outputs,
        if not speficied, they are computed with the reference evaluator
    :param data_set: to have multiple tests with the same model
    :return: test folder
    """
    if outputs is None:
        ref = CReferenceEvaluator(model)
        outputs = ref.run(None, dict(zip([i.name for i in model.graph.input], inputs)))

    input_protos = [
        from_array_extended(inp, name)
        for name, inp in zip([i.name for i in model.graph.input], inputs)
    ]
    output_protos = [
        from_array_extended(out, name)
        for name, out in zip([i.name for i in model.graph.output], outputs)
    ]
    model_bytes = model.SerializeToString()

    path = os.path.join(folder, test_name)
    if not os.path.exists(path):
        os.makedirs(path)
    model_file = os.path.join(path, "model.onnx")
    with open(model_file, "wb") as f:
        f.write(model_bytes)

    sub_path = os.path.join(path, f"test_data_set_{data_set}")
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    for i, t in enumerate(input_protos):
        with open(os.path.join(sub_path, f"input_{i}.pb"), "wb") as f:
            f.write(t.SerializeToString())
    for i, t in enumerate(output_protos):
        with open(os.path.join(sub_path, f"output_{i}.pb"), "wb") as f:
            f.write(t.SerializeToString())
    return path
