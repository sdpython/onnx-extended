import os
import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from onnx import ModelProto, TensorProto, load
from onnx.reference.op_run import to_array_extended
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


class TestRun:
    """
    Loads a test saved by :func:`save_for_benchmark_or_test`.

    :param folder: test folder

    It has the following attributes:

    * `folder: str`
    * `proto: ModelProto`
    * `datasets: Dict[int, List[Tuple[int, np.array]]]`
    """

    def __init__(self, folder: str):
        self.folder = folder
        self._load()

    @staticmethod
    def _load_list(
        io_list: List[Tuple[int, str]], asarray: bool = True
    ) -> List[Tuple[int, TensorProto]]:
        new_list = []
        for i, name in io_list:
            with open(name, "rb") as f:
                content = f.read()
            pb = TensorProto()
            pb.ParseFromString(content)
            if asarray:
                new_list.append((i, to_array_extended(pb)))
            else:
                new_list.append((i, pb))
        return new_list

    def _load(self):
        "Loads the test."
        model = os.path.join(self.folder, "model.onnx")
        if not os.path.exists(model):
            raise FileNotFoundError(f"Unable to find {model!r}.")
        with open(model, "rb") as f:
            self.proto = load(f)

        datasets = []
        for sub in os.listdir(self.folder):
            if not sub.startswith("test_data_set_"):
                continue
            index = int(sub.replace("test_data_set_", ""))
            subf = os.path.join(self.folder, sub)
            ios = os.listdir(subf)
            inputs = []
            outputs = []
            for name in ios:
                fullname = os.path.join(subf, name)
                if name.startswith("input_"):
                    ii = int(os.path.splitext(name)[0][6:])
                    inputs.append((ii, os.path.join(subf, fullname)))
                elif name.startswith("output_"):
                    io = int(os.path.splitext(name)[0][7:])
                    outputs.append((io, os.path.join(subf, fullname)))
                else:
                    raise RuntimeError(f"Unexpected file {name!r} in {subf!r}.")

            inputs.sort()
            outputs.sort()
            inputs_pb = self._load_list(inputs)
            outputs_pb = self._load_list(outputs)
            datasets.append((index, (inputs_pb, outputs_pb)))

        self.datasets = dict(datasets)

    def __len__(self) -> int:
        "Returns the number of loaded datasets."
        return len(self.datasets)

    @property
    def input_names(self):
        "Returns the input names of the model."
        return [i.name for i in self.proto.graph.input]

    @property
    def output_names(self):
        "Returns the output names of the model."
        return [i.name for i in self.proto.graph.output]

    def test(
        self,
        f_build: Callable[[ModelProto], Any],
        f_run: Callable[[Any, Dict[str, np.array]], List[np.array]],
        index: int = 0,
        exc: bool = True,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> Optional[Dict[str, Tuple[float, float, str]]]:
        """
        Runs the tests.

        :param f_build: function to call to build the inference class
        :param f_run: function to call to run the inference
        :param index: test index
        :param exc: raises an exception if the verification fails
        :param atol: absolute tolerance
        :param rtol: relative tolerance
        :return: list of results with discrepancies,
            the absolute error, the relative one and a reason for the failure
        """
        input_names = self.input_names
        output_names = self.output_names
        inputs, outputs = self.datasets[index]
        feeds = {}
        for ii, tensor in inputs:
            feeds[input_names[ii]] = tensor

        rt = f_build(self.proto)
        results = f_run(rt, feeds)

        d_outputs = dict(outputs)
        checks = {}
        for i, res in enumerate(results):
            name = output_names[i]
            expected = d_outputs[i]
            if expected.dtype != res.dtype:
                reason = (
                    f"Type mismatch for output {i}, {expected.dtype} != {res.dtype}"
                )
                checks[name] = (np.nan, np.nan, reason)
                continue
            if expected.shape != res.shape:
                reason = (
                    f"Shape mismatch for output {i}, {expected.shape} != {res.shape}"
                )
                checks[name] = (np.nan, np.nan, reason)
                continue
            expected64 = expected.astype(np.float64)
            res64 = res.astype(np.float64)
            diff = np.abs(res64 - expected64)
            max_diff = diff.max()
            rel_diff = (diff / np.maximum(expected, max(1e-15, atol))).max()

            if max_diff > atol:
                reason = f"Discrepancies for output {i}, atol={max_diff}"
                checks[name] = (max_diff, rel_diff, reason)
                continue
            if rel_diff > rtol:
                reason = f"Discrepancies for output {i}, rtol={rel_diff}"
                checks[name] = (max_diff, rel_diff, reason)
                continue

        if len(checks) == 0:
            return None
        if exc:
            raise AssertionError(
                f"Results do not match the expected value:"
                f"\n{pprint.pformat(checks)}"
            )
        return checks
