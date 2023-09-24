import json
import os
import pprint
import subprocess
import time
import sys
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve
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

    def bench(
        self,
        f_build: Callable[[ModelProto], Any],
        f_run: Callable[[Any, Dict[str, np.array]], List[np.array]],
        index: int = 0,
        warmup: int = 5,
        repeat: int = 10,
    ) -> Dict[str, Union[float, Dict[str, Tuple[int, ...]]]]:
        """
        Runs the model on the given inputs.

        :param f_build: function to call to build the inference class
        :param f_run: function to call to run the inference
        :param index: test index to measure
        :param warmup: number of iterations to run before
            starting to measure the model
        :param repeat: number of iterations to measure
        :return: dictionary with many metrics,
            any metric endings with `"_time"` is a duration
        """
        str_type = {
            np.float16: "float16",
            np.float32: "float32",
            np.float64: "float64",
            np.int8: "int8",
            np.int16: "int16",
            np.int32: "int32",
            np.int64: "int64",
            np.uint8: "uint8",
            np.uint16: "uint16",
            np.uint32: "uint32",
            np.uint64: "uint64",
        }

        stats = {}
        begin = time.perf_counter()
        rt = f_build(self.proto)
        stats["build_time"] = time.perf_counter() - begin

        shapes = {}
        dtypes = {}
        input_names = self.input_names
        inputs = self.datasets[index][0]
        feeds = {}
        input_size = 0
        for ii, tensor in inputs:
            feeds[input_names[ii]] = tensor
            shapes[input_names[ii]] = tensor.shape
            dtypes[input_names[ii]] = str_type.get(tensor.dtype, str(tensor.dtype))
            input_size += np.prod(tensor.shape)
        stats["shapes"] = shapes
        stats["dtypes"] = dtypes
        stats["input_size"] = int(input_size)

        begin = time.perf_counter()
        for _ in range(warmup):
            f_run(rt, feeds)
        stats["warmup_time"] = time.perf_counter() - begin
        stats["warmup"] = warmup
        stats["name"] = self.folder
        stats["index"] = index

        ts = []
        for i in range(repeat):
            begin = time.perf_counter()
            f_run(rt, feeds)
            ts.append(time.perf_counter() - begin)

        stats["repeat"] = repeat
        stats["avg_time"] = float(np.array(ts).mean())
        stats["min_time"] = float(np.array(ts).min())
        stats["max_time"] = float(np.array(ts).max())

        ts.sort()
        if repeat > 4:
            stats["max1_time"] = ts[-2]
            stats["min1_time"] = ts[1]
        return stats


def _run_cmd(args: List[str]) -> Tuple[str, str]:
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={"PYTHONPATH": ""}
    )
    st = StringIO()
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            out = output.rstrip()
            st.write(out + "\n")
    p.poll()
    p.stdout.close()
    return st.getvalue()


def bench_virtual(
    test_path: str,
    virtual_path: str,
    runtimes: Union[List[str], str] = "ReferenceEvaluator",
    index: int = 0,
    warmup: int = 5,
    repeat: int = 10,
    modules: Optional[List[Dict[str, str]]] = None,
    verbose: int = 0,
    save_as_dataframe: Optional[str] = None,
) -> List[Dict[str, Union[float, Dict[str, Tuple[int, ...]]]]]:
    """
    Runs the same benchmark over different
    versions of the same packages in a virtual environment.

    :param test_path: test path
    :param virtual_path: path to the virtual environment
    :param runtimes: runtimes to measure
        (ReferenceEvaluation, CReferenceEvaluator, onnxruntime)
    :param index: test index to measure
    :param warmup: number of iterations to run before
        starting to measure the model
    :param repeat: number of iterations to measure
    :param modules: modules to install, example:
        `modules=[{"onnxruntime": "1.16.0", "onnx": "1.15.0"}]`
    :param verbose: verbosity
    :param save_as_dataframe: saves as dataframe
    :return: list of statistics
    """
    exe = os.path.join(virtual_path, "bin", "python")
    if not os.path.exists(exe):
        if verbose > 0:
            print(f"[bench_virtual] create the virtual environment in {virtual_path!r}")
        out = _run_cmd([sys.executable, "-m", "venv", virtual_path])
        if verbose > 2:
            print(out)
        if not os.path.exists(exe):
            raise RuntimeError(f"The virtual environment was not created:\n{out}")
        get_pip = os.path.join(virtual_path, "get_pip.py")
        if not os.path.exists(get_pip):
            if verbose > 2:
                print("[bench_virtual] install pip")
            urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        out = _run_cmd([exe, get_pip])
        if verbose > 2:
            print(out)

    if modules is None:
        ext = "https://github.com/sdpython/onnx-extended.git"
        modules = [
            {"onnxruntime": "1.16.0", "onnx": None, "onnx-extended": f"git+{ext}"},
            {"onnxruntime": "1.13.1", "onnx": None, "onnx-extended": f"git+{ext}"},
        ]
    if isinstance(runtimes, str):
        runtimes = [runtimes]

    # mandatory packages
    for name in ["setuptools", "wheel", "pybind11", "cython", "tomli", "packaging"]:
        out = _run_cmd([exe, "-m", "pip", "install", name])
        if verbose > 2:
            print(out)

    # packages defined by the user
    obs = []
    for i, conf in enumerate(modules):
        if verbose > 2:
            print("-------------------------------------------------------")
        if verbose > 0:
            print(f"[bench_virtual] {i+1}/{len(modules)}:{conf}")

        for k, v in conf.items():
            if verbose > 1:
                print(f"[bench_virtual] uninstall {k}")
            out = _run_cmd([exe, "-m", "pip", "uninstall", "-y", k])
            if verbose > 2:
                print(out)
            if verbose > 2:
                print(out)
            if verbose > 1:
                print(f"[bench_virtual] install {k}: {v or 'upgrade'}")
            if v is None:
                out = _run_cmd([exe, "-m", "pip", "install", k, "--upgrade"])
                if verbose > 2:
                    print(out)
            elif v.startswith("git"):
                out = _run_cmd([exe, "-m", "pip", "install", v])
                if verbose > 2:
                    print(out)
            else:
                out = _run_cmd([exe, "-m", "pip", "install", f"{k}=={v}"])
                if verbose > 2:
                    print(out)
            if verbose > 1:
                print(f"[bench_virtual] check {k} is installed")
            out = _run_cmd(
                [exe, "-c", f"import {k.replace('-', '_')} as m;print(m.__file__)"]
            )
            if verbose > 2:
                print(out)
            if "Error" in out:
                raise RuntimeError(out)

        for rt in runtimes:
            if verbose > 1:
                print(f"[bench_virtual] run with {rt}")
            out = _run_cmd(
                [
                    exe,
                    "-m",
                    "onnx_extended.tools.run_onnx_main",
                    "-p",
                    test_path,
                    "-r",
                    str(repeat),
                    "-w",
                    str(warmup),
                    "-e",
                    rt,
                ]
            )
            if "Traceback" in out:
                raise RuntimeError(out)
            try:
                js = json.loads(out)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(f"Unable to decode {out!r}") from e
            if verbose > 2:
                print("[bench_virtual] final results")
                print(js)
            obs.append(js)

    if save_as_dataframe:
        import pandas

        df = pandas.DataFrame(obs)
        df.to_csv(save_as_dataframe, index=False)
        return df
    return obs
