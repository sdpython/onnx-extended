"""
Evaluate different implementation of TreeEnsemble
=================================================

This is a simplified bencharmk to compare TreeEnsemble implementations.

Sparse Data
+++++++++++
"""
import logging
import os
import subprocess
import multiprocessing
from typing import Any, Iterator, Tuple
import warnings
import numpy
import onnx
from onnx import ModelProto, TensorProto
from onnx.helper import make_attribute, make_graph, make_model, make_tensor_value_info
from pandas import DataFrame
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from onnxruntime import InferenceSession, SessionOptions
from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
)
from onnx_extended.tools.onnx_nodes import multiply_tree
from onnx_extended.validation.cpu._validation import dense_to_sparse_struct

# from onnx_extended.plotting.benchmark import hhistograms
from onnx_extended.args import get_parsed_args
from onnx_extended.ext_test_case import unit_test_going
from onnx_extended.ext_test_case import measure_time

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

script_args = get_parsed_args(
    "plot_op_tree_ensemble_sparse",
    description=__doc__,
    scenarios={
        "SHORT": "short optimization (default)",
        "LONG": "test more options",
        "CUSTOM": "use values specified by the command line",
    },
    sparsity=(0.99, "input sparsity"),
    n_features=(2 if unit_test_going() else 500, "number of features to generate"),
    n_trees=(3 if unit_test_going() else 10, "number of trees to train"),
    max_depth=(2 if unit_test_going() else 10, "max_depth"),
    batch_size=(1000 if unit_test_going() else 1000, "batch size"),
    warmup=1 if unit_test_going() else 3,
    parallel_tree=(40, "values to try for parallel_tree"),
    parallel_tree_N=(64, "values to try for parallel_tree_N"),
    parallel_N=(50, "values to try for parallel_N"),
    batch_size_tree=(4, "values to try for batch_size_tree"),
    batch_size_rows=(4, "values to try for batch_size_rows"),
    use_node3=(0, "values to try for use_node3"),
    expose="",
    n_jobs=("-1", "number of jobs to train the RandomForestRegressor"),
)


################################
# Training a model
# ++++++++++++++++


def train_model(
    batch_size: int, n_features: int, n_trees: int, max_depth: int, sparsity: float
) -> Tuple[str, numpy.ndarray, numpy.ndarray]:
    filename = (
        f"plot_op_tree_ensemble_sparse-f{n_features}-{n_trees}-"
        f"d{max_depth}-s{sparsity}.onnx"
    )
    if not os.path.exists(filename):
        X, y = make_regression(
            batch_size + max(batch_size, 2 ** (max_depth + 1)),
            n_features=n_features,
            n_targets=1,
        )
        mask = numpy.random.rand(*X.shape) <= sparsity
        X[mask] = 0
        X, y = X.astype(numpy.float32), y.astype(numpy.float32)

        print(f"Training to get {filename!r} with X.shape={X.shape}")
        # To be faster, we train only 1 tree.
        model = RandomForestRegressor(
            1, max_depth=max_depth, verbose=2, n_jobs=int(script_args.n_jobs)
        )
        model.fit(X[:-batch_size], y[:-batch_size])
        onx = to_onnx(model, X[:1])

        # And wd multiply the trees.
        node = multiply_tree(onx.graph.node[0], n_trees)
        onx = make_model(
            make_graph([node], onx.graph.name, onx.graph.input, onx.graph.output),
            domain=onx.domain,
            opset_imports=onx.opset_import,
        )

        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
    else:
        X, y = make_regression(batch_size, n_features=n_features, n_targets=1)
        mask = numpy.random.rand(*X.shape) <= sparsity
        X[mask] = 0
        X, y = X.astype(numpy.float32), y.astype(numpy.float32)
    Xb, yb = X[-batch_size:].copy(), y[-batch_size:].copy()
    return filename, Xb, yb


def measure_sparsity(x):
    f = x.flatten()
    return float((f == 0).astype(numpy.int64).sum()) / float(x.size)


batch_size = script_args.batch_size
n_features = script_args.n_features
n_trees = script_args.n_trees
max_depth = script_args.max_depth
sparsity = script_args.sparsity
warmup = script_args.warmup

print(f"batch_size={batch_size}")
print(f"n_features={n_features}")
print(f"n_trees={n_trees}")
print(f"max_depth={max_depth}")
print(f"sparsity={sparsity}")
print(f"warmup={warmup}")

filename, Xb, yb = train_model(batch_size, n_features, n_trees, max_depth, sparsity)

print(f"Xb.shape={Xb.shape}")
print(f"yb.shape={yb.shape}")
print(f"measured sparsity={measure_sparsity(Xb)}")

#############################################
# Implementations
# +++++++++++++++


def compile_tree(
    llc_exe: str,
    filename: str,
    onx: ModelProto,
    batch_size: int,
    tree_tile_size: int = 8,
    verbose: int = 0,
) -> str:
    """
    Compiles a tree with `TreeBeard <https://github.com/asprasad/treebeard>`_.
    """
    if verbose:
        print("[compile_tree] import treebeard")
    import treebeard

    if verbose:
        print(
            f"[compile_tree] treebeard set options, "
            f"batch_size={batch_size}, tree_tile_size={tree_tile_size}"
        )
    compiler_options = treebeard.CompilerOptions(batch_size, tree_tile_size)

    compiler_options.SetNumberOfCores(multiprocessing.cpu_count())
    compiler_options.SetMakeAllLeavesSameDepth(1)
    compiler_options.SetReorderTreesByDepth(True)
    assert 8 < batch_size
    compiler_options.SetPipelineWidth(8)

    if verbose:
        print(f"[compile_tree] write filename={filename!r}")

    # let's remove nodes_hitrates to avoid a warning before saving the model
    for node in onx.graph.node:
        if node.op_type == "TreeEnsembleRegressor":
            found = -1
            for i in range(len(node.attribute)):
                if node.attribute[i].name == "nodes_hitrates":
                    found = i
            if found >= 0:
                del node.attribute[found]
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())

    onnx_model_path = os.path.abspath(filename)
    if verbose:
        print(
            f"[compile_tree] treebeard context with onnx_model_path={onnx_model_path!r}"
        )
    tbContext = treebeard.TreebeardContext(onnx_model_path, "", compiler_options)
    tbContext.SetRepresentationType("sparse")
    tbContext.SetInputFiletype("onnx_file")

    llvm_file_path = f"{os.path.splitext(onnx_model_path)[0]}.ll"
    if verbose:
        print(f"[compile_tree] LLVM dump into {llvm_file_path!r}")
    error = tbContext.DumpLLVMIR(llvm_file_path)
    if error:
        raise RuntimeError(
            f"Failed to dump LLVM IR in {llvm_file_path!r}, error={error}."
        )
    if not os.path.exists(llvm_file_path):
        raise FileNotFoundError(f"Unable to find {llvm_file_path!r}.")

    # Run LLC
    asm_file_path = f"{os.path.splitext(onnx_model_path)[0]}.s"
    if verbose:
        print(f"[compile_tree] llc={llc_exe!r}")
        print(f"[compile_tree] run LLC into {llvm_file_path!r}")
    subprocess.run(
        [
            llc_exe,
            llvm_file_path,
            "-O3",
            "-march=x86-64",
            "-mcpu=native",
            "--relocation-model=pic",
            "-o",
            asm_file_path,
        ]
    )

    # Run CLANG
    so_file_path = f"{os.path.splitext(onnx_model_path)[0]}.so"
    if verbose:
        print(f"[compile_tree] run clang into {so_file_path!r}")
    subprocess.run(
        ["clang", "-shared", asm_file_path, "-fopenmp=libomp", "-o", so_file_path]
    )
    if verbose:
        print("[compile_tree] done.")
    return so_file_path


def make_ort_assembly_session(onx: ModelProto, batch_size: int) -> Any:
    from onnxruntime import InferenceSession, SessionOptions
    from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs as lib_tuto

    llc_exe = os.environ.get("TEST_LLC_EXE", "SKIP")
    if llc_exe == "SKIP":
        warnings.warn("Unable to find environment variable 'TEST_LLC_EXE'.")
        return None

    filename = "plot_op_tree_ensemble_implementation.onnx"
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    onx = onnx.load(filename)
    assembly_name = compile_tree(
        llc_exe,
        filename,
        onx,
        batch_size,
        verbose=1 if __name__ == "__main__" else 0,
    )

    # assembly
    for node in onx.graph.node:
        if node.op_type == "TreeEnsembleRegressor":
            node.op_type = "TreeEnsembleAssemblyRegressor"
            node.domain = "onnx_extented.ortops.tutorial.cpu"
            del node.attribute[:]
            new_add = make_attribute("assembly", assembly_name)
            node.attribute.append(new_add)

    d = onx.opset_import.add()
    d.domain = "onnx_extented.ortops.tutorial.cpu"
    d.version = 1

    r = lib_tuto()
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    sess_assembly = InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    return sess_assembly


def transform_model(model, use_sparse=False, **kwargs):
    onx = ModelProto()
    onx.ParseFromString(model.SerializeToString())
    att = get_node_attribute(onx.graph.node[0], "nodes_modes")
    modes = ",".join(map(lambda s: s.decode("ascii"), att.strings)).replace(
        "BRANCH_", ""
    )
    if use_sparse and "new_op_type" not in kwargs:
        kwargs["new_op_type"] = "TreeEnsembleRegressorSparse"
    if use_sparse:
        # with sparse tensor, missing value means 0
        att = get_node_attribute(onx.graph.node[0], "nodes_values")
        thresholds = numpy.array(att.floats, dtype=numpy.float32)
        missing_true = (thresholds >= 0).astype(numpy.int64)
        kwargs["nodes_missing_value_tracks_true"] = missing_true
    new_onx = change_onnx_operator_domain(
        onx,
        op_type="TreeEnsembleRegressor",
        op_domain="ai.onnx.ml",
        new_op_domain="onnx_extented.ortops.optim.cpu",
        nodes_modes=modes,
        **kwargs,
    )
    if use_sparse:
        del new_onx.graph.input[:]
        new_onx.graph.input.append(
            make_tensor_value_info("X", TensorProto.FLOAT, (None,))
        )
    return new_onx


def enumerate_implementations(
    onx: ModelProto, X: "Tensor", **kwargs  # noqa: F821
) -> Iterator[Tuple[str, Any, "Tensor"]]:  # noqa: F821
    """
    Creates all the InferenceSession.
    """
    providers = ["CPUExecutionProvider"]
    yield (
        "ort",
        InferenceSession(onx.SerializeToString(), providers=providers),
        X,
    )

    r = get_ort_ext_libs()
    opts = SessionOptions()
    if r is not None:
        opts.register_custom_ops_library(r[0])

    tr = transform_model(onx)
    yield (
        "custom",
        InferenceSession(tr.SerializeToString(), opts, providers=providers),
        X,
    )

    tr = transform_model(onx, **kwargs)
    yield (
        "cusopt",
        InferenceSession(tr.SerializeToString(), opts, providers=providers),
        X,
    )

    Xsp = dense_to_sparse_struct(X)
    tr = transform_model(onx, use_sparse=True, **kwargs)
    yield (
        "sparse",
        InferenceSession(tr.SerializeToString(), opts, providers=providers),
        Xsp,
    )

    sess = make_ort_assembly_session(onx, batch_size=X.shape[0])
    yield ("assembly", sess, X)


kwargs = dict(
    parallel_tree=40,
    parallel_tree_N=128,
    parallel_N=50,
    batch_size_tree=4,
    batch_size_rows=4,
    use_node3=0,
)

onx = onnx.load(filename)
sessions = []

print("----- warmup")
for name, sess, tensor in enumerate_implementations(onx, Xb, **kwargs):
    if sess is None:
        continue
    sessions.append((name, sess, tensor))
    print(f"run {name!r}")
    feeds = {"X": tensor}
    sess.run(None, feeds)
print("done.")


#############################################
# Benchmark implementations
# +++++++++++++++++++++++++


data = []

print("----- measure time")
for name, sess, tensor in sessions:
    print(f"run {name!r}")
    feeds = {"X": tensor}
    obs = measure_time(
        lambda: sess.run(None, feeds),
        repeat=script_args.repeat,
        number=script_args.number,
        warmup=script_args.warmup,
    )
    obs["name"] = name
    data.append(obs)
print("done.")

df = DataFrame(data)
print(df)

####################################
# Plots.
print(df.columns)

ax = (
    df[["name", "average"]]
    .set_index("name")
    .plot.barh(
        title="Compare implementations of TreeEnsemble",
        xerr=[df["min_exec"], df["max_exec"]],
    )
)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig("plot_tree_ensemble_implementations.png")
