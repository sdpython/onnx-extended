import cProfile
import datetime
import io
import logging
import pstats
import time
import warnings
from typing import Any, Dict, List, Optional
import numpy as np
from onnx import ModelProto
from onnx.helper import make_model, make_graph


def create_decision_tree(n_features: int = 100, max_depth: int = 14) -> ModelProto:
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor
    from skl2onnx import to_onnx

    logging.getLogger("skl2onnx").setLevel(logging.ERROR)

    # from ..tools.onnx_nodes import onnx2string
    X, y = make_regression(2 ** (max_depth + 1), n_features=n_features, n_targets=1)
    X, y = X.astype(np.float32), y.astype(np.float32)
    batch_size = 2**max_depth
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X[:-batch_size], y[:-batch_size])
    onx = to_onnx(model, X[:1])
    return onx


class Engine:
    """
    Implements a common interface to the different ways to
    run the inference.
    """

    def __init__(self, name: str, sess: Any):
        self.name = name
        self.sess_ = sess

    def run(self, unused: Any, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.sess_.run(unused, feeds)


class EngineCython(Engine):
    """
    Same interface as InferenceSession but for
    :class:`OrtSession <onnx_extended.ortcy.wrap.ortinf.OrtSession>`.
    """

    def run(self, unused: Any, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.sess_.run(list(feeds.values()))


def create_engine(name: str, onx: ModelProto, feeds: Dict[str, np.ndarray]) -> Engine:
    """
    Creates engines to benchmark a random forest.

    :param name: name of the engine, see below
    :param onx: the model
    :param feeds: the inputs
    :return: an instance of :class:`Engine`

    Possible choices:

    * `onnxruntime`: simple onnxruntime.InferenceSession
    * `onnxruntime-customop`: onnxruntime.InferenceSession
      with a custom implementation for the trees
    * `CReferenceEvaluator`: :class:`onnx_extended.reference.CReferenceEvaluator`
    * `cython`: cython wrapper for the onnxruntime shared libraries
    * `cython-customop`: cython wrapper for the onnxruntime shared libraries
      with a custom implementation for the trees
    """
    if name == "onnxruntime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from onnxruntime import InferenceSession

        eng = Engine(
            name,
            InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            ),
        )

    elif name == "CReferenceEvaluator":
        from ..reference import CReferenceEvaluator

        eng = Engine(name, CReferenceEvaluator(onx))
    elif name == "onnxruntime-customops":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from onnxruntime import InferenceSession, SessionOptions
        from ..ortops.optim.cpu import get_ort_ext_libs
        from ..ortops.optim.optimize import (
            change_onnx_operator_domain,
            get_node_attribute,
        )

        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleRegressor",
            op_domain="ai.onnx.ml",
            new_op_domain="onnx_extended.ortops.optim.cpu",
            nodes_modes=",".join(
                [
                    s.decode("ascii")
                    for s in get_node_attribute(
                        onx.graph.node[0], "nodes_modes"
                    ).strings
                ]
            ),
        )

        so = SessionOptions()
        so.register_custom_ops_library(get_ort_ext_libs()[0])
        eng = Engine(
            name,
            InferenceSession(
                onx2.SerializeToString(), so, providers=["CPUExecutionProvider"]
            ),
        )

    elif name == "cython":
        from ..ortcy.wrap.ortinf import OrtSession

        eng = EngineCython(name, OrtSession(onx.SerializeToString()))

    elif name == "cython-customops":
        from ..ortcy.wrap.ortinf import OrtSession
        from ..ortops.optim.cpu import get_ort_ext_libs
        from ..ortops.optim.optimize import (
            change_onnx_operator_domain,
            get_node_attribute,
        )

        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleRegressor",
            op_domain="ai.onnx.ml",
            new_op_domain="onnx_extended.ortops.optim.cpu",
            nodes_modes=",".join(
                [
                    s.decode("ascii")
                    for s in get_node_attribute(
                        onx.graph.node[0], "nodes_modes"
                    ).strings
                ]
            ),
        )
        eng = EngineCython(
            name, OrtSession(onx2.SerializeToString(), custom_libs=get_ort_ext_libs())
        )
    else:
        raise NotImplementedError(f"Unable to create engin for name={name!r}.")

    return eng


def bench_trees(
    max_depth: int = 14,
    n_estimators: int = 100,
    n_features: int = 100,
    batch_size=10000,
    number: int = 10,
    warmup: int = 2,
    verbose: int = 0,
    engine_names: Optional[List[str]] = None,
    repeat: int = 2,
    profile: bool = False,
) -> List[Dict[str, Any]]:
    """
    Measures the performances of the different implements of the TreeEnsemble.

    :param max_depth: depth of tree
    :param n_estimators: number of trees in the forest
    :param n_features: number of features
    :param batch_size: batch size
    :param number: number of calls to measure
    :param warmup: number of calls before starting the measure
    :param verbose: verbosity
    :param engine_names: see below
    :param repeat: number of times to repeat the measure
    :param profile: run a profiler as well
    :return: list of observations

    Possible choices:

    * `onnxruntime`: simple onnxruntime.InferenceSession
    * `onnxruntime-customop`: onnxruntime.InferenceSession
      with a custom implementation for the trees
    * `CReferenceEvaluator`: :class:`onnx_extended.reference.CReferenceEvaluator`
    * `cython`: cython wrapper for the onnxruntime shared libraries
    * `cython-customop`: cython wrapper for the onnxruntime shared libraries
      with a custom implementation for the trees
    """
    from ..tools.onnx_nodes import multiply_tree

    now = lambda: datetime.datetime.now().time()  # noqa: E731

    if n_features == 100 and max_depth == 14:
        if verbose > 0:
            print(f" [bench_trees] {now()} import tree")
        from ._tree_d14_f100 import tree_d14_f100

        tree = tree_d14_f100()
    else:
        if verbose > 0:
            print(f" [bench_trees] {now()} create tree")
        tree = create_decision_tree(n_features=n_features, max_depth=max_depth)

    if verbose > 0:
        print(f" [bench_trees] {now()} create forest with {n_estimators} trees")
    onx2 = multiply_tree(tree.graph.node[0], n_estimators)
    new_tree = make_model(
        make_graph([onx2], tree.graph.name, tree.graph.input, tree.graph.output),
        domain=tree.domain,
        opset_imports=tree.opset_import,
        ir_version=tree.ir_version,
    )

    if verbose > 0:
        print(
            f" [bench_trees] {now()} modelsize "
            f"{float(len(new_tree.SerializeToString()))/2**10:1.3f} Kb"
        )
        print(f" [bench_trees] {now()} create datasets")

    from sklearn.datasets import make_regression

    X, _ = make_regression(batch_size, n_features=n_features, n_targets=1)
    feeds = {"X": X.astype(np.float32)}

    # self.assertRaise(lambda: multiply_tree(onx, 2), TypeError)
    # onx2 = multiply_tree(onx.graph.node[0], 2)
    if verbose > 0:
        print(f" [bench_trees] {now()} create engines")
    if engine_names is None:
        engine_names = [
            "onnxruntime",
            "CReferenceEvaluator",
            "onnxruntime-customops",
            "cython",
            "cython-customops",
        ]
    engines = {}
    for name in engine_names:
        if verbose > 1:
            print(f" [bench_trees] {now()} create engine {name!r}")
        engines[name] = create_engine(name, new_tree, feeds)

    if verbose > 0:
        print(f" [bench_trees] {now()} benchmark")

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    results = []
    for r in range(repeat):
        for name, engine in engines.items():
            if verbose > 1:
                print(f" [bench_trees] {now()} test {name!r} warmup...")

            for _ in range(warmup):
                engine.run(None, feeds)

            if verbose > 1:
                print(f" [bench_trees] {now()} test {name!r} benchmark...")

            begin = time.perf_counter()
            for _i in range(number):
                feeds["X"] += feeds["X"] * np.float32(np.random.random() / 1000)
                engine.run(None, feeds)
            duration = time.perf_counter() - begin

            if verbose > 1:
                print(
                    f" [bench_trees] {now()} test {name!r} "
                    f"duration={float(duration) / number}"
                )
            results.append(
                dict(
                    name=name,
                    repeat=r,
                    duration=float(duration) / number,
                    n_estimators=n_estimators,
                    number=number,
                    n_features=n_features,
                    max_depth=max_depth,
                    batch_size=batch_size,
                )
            )
    if profile:
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    return results
