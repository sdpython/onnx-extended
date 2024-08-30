from itertools import permutations
from typing import Any, Dict, Iterable, List, Optional, Union
import numpy
from onnx import helper, ModelProto, TensorProto
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession
from ...ext_test_case import measure_time
from .einsum_config import DEFAULT_OPSET, DEFAULT_IR_VERSION
from .einsum_impl import decompose_einsum_equation, apply_einsum_sequence


def _measure_time(
    stmt: Any,
    *x: List[numpy.ndarray],
    repeat: int = 5,
    number: int = 5,
    div_by_number: bool = True,
    first_run: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Union[str, float]]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string
    :param *x: inputs
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param div_by_number: divide by the number of executions
    :param first_run: if True, runs the function once before measuring
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    See `Timer.repeat
    <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if first_run:
        try:
            stmt(*x)
        except RuntimeError as e:
            raise RuntimeError(f"{type(x)}-{getattr(x, 'dtype', '?')}") from e

    def fct():
        stmt(*x)

    if first_run:
        fct()

    return measure_time(
        fct,
        context={},
        repeat=repeat,
        number=number,
        div_by_number=div_by_number,
        max_time=max_time,
    )


def _make_einsum_model(equation: str, opset: int = DEFAULT_OPSET) -> ModelProto:
    inputs = equation.split("->")[0].split(",")

    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid("", opset)],
        ir_version=DEFAULT_IR_VERSION,
        producer_name="onnx_extended",
        producer_version="0.1",
        graph=helper.make_graph(
            name="einsum_test",
            inputs=[
                helper.make_tensor_value_info("X%d" % i, TensorProto.FLOAT, None)
                for i in range(len(inputs))
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            nodes=[
                helper.make_node(
                    "Einsum",
                    ["X%d" % i for i in range(len(inputs))],
                    ["Y"],
                    equation=equation,
                )
            ],
        ),
    )
    return model


def _make_inputs(equation, shapes):
    inputs = equation.split("->")[0].split(",")
    dims = [len(i) for i in inputs]

    if isinstance(shapes, int):
        N = shapes
        shapes = [(N,) * le for le in dims]
    else:
        assert len(shapes) == len(
            inputs
        ), f"Unexpected number of shapes {shapes!r} with equation {equation!r}."
    inputs = [numpy.random.randn(*sh) for sh in shapes]
    return [i.astype(numpy.float32) for i in inputs]


def einsum_benchmark(
    equation: str = "abc,cd->abd",
    shape: int = 30,
    perm: bool = False,
    runtime: str = "python",
    use_tqdm: bool = False,
    number: int = 5,
    repeat: int = 5,
    opset=DEFAULT_OPSET,
) -> Iterable[Dict[str, Union[str, float]]]:
    """
    Investigates whether or not the decomposing einsum is faster.

    :param equation: einsum equation to test
    :param shape: an integer (all dimension gets the same size) or
        a list of shapes in a string separated with `;`)
    :param perm: check on permutation or all letter permutations
    :param runtime: a string among 'numpy', 'python', 'onnxruntime'
    :param use_tqdm: show progress
    :param number: usual parameter to measure a function
    :param repeat: usual parameter to measure a function
    :param opset: target opset
    :return: list of dictionaries as an iterator
    """
    scenarios = []
    if isinstance(shape, list) and all(isinstance(t, int) for t in shape):
        shape_list = shape
    else:
        shape_list = [shape]

    if perm:
        assert equation.lower() == equation, (
            "Only equations with lower letters are allowed but equation %r "
            "is not." % equation
        )
        letters = list(
            sorted(set(c for c in equation if "a" <= c < "z" or "A" <= c < "Z"))
        )
        for p in permutations(letters):
            replace = {d: c for c, d in zip(letters, p)}
            eq = equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            for dec in ["einsum", "dec"]:
                for sh in shape_list:
                    scenarios.append((eq, runtime, dec, sh))
    else:
        for dec in ["einsum", "dec"]:
            for sh in shape_list:
                scenarios.append((equation, runtime, dec, sh))

    if use_tqdm:
        from tqdm import tqdm

        loop = tqdm(scenarios)
    else:
        loop = scenarios

    for eq, rt, dec, sh in loop:
        inputs = _make_inputs(equation, sh)

        if dec == "dec":
            seq = decompose_einsum_equation(eq, strategy="numpy", clean=True)
        else:
            seq = None

        if rt == "numpy":
            if dec == "einsum":
                fct = lambda *x, eq=eq: numpy.einsum(eq, *x, optimize=True)
            else:
                fct = lambda *x, seq=seq: apply_einsum_sequence(seq, *x)
        elif rt == "onnxruntime":
            if dec == "einsum":
                onx = _make_einsum_model(equation, opset=opset)
            else:
                assert seq is not None, "seq cannot be None."
                onx = seq.to_onnx(
                    "Y", *["X%d" % i for i in range(len(inputs))], opset=opset
                )
            sess = InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            fct = lambda *x, se=sess: se.run(
                None, {"X%d" % i: v for i, v in enumerate(x)}
            )
        elif rt == "python":
            if dec == "einsum":
                onx = _make_einsum_model(equation, opset=opset)
            else:
                assert seq is not None, "seq must not be None."
                onx = seq.to_onnx(
                    "Y", *["X%d" % i for i in range(len(inputs))], opset=opset
                )
            oinf = ReferenceEvaluator(onx)
            fct = lambda *x, oi=oinf: oi.run(
                None, {"X%d" % i: v for i, v in enumerate(x)}
            )
        else:
            raise ValueError(f"Unexpected runtime {rt!r}.")

        res = _measure_time(fct, *inputs, repeat=repeat, number=number)
        res["rt"] = rt
        res["dec"] = dec
        res["eq"] = eq
        res["shapes"] = ";".join(map(str, [m.shape for m in inputs])).replace(" ", "")
        yield res
