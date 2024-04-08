"""
.. _l-plot-optim-tree-ensemble-sparse:

TreeEnsemble, dense, and sparse
===============================

The example benchmarks the sparse implementation for TreeEnsemble.
The default set of optimized parameters is very short and is meant to be executed
fast. Many more parameters can be tried.

::

    python plot_op_tree_ensemble_sparse --scenario=LONG

To change the training parameters:

::

    python plot_op_tree_ensemble_sparse.py
        --n_trees=100
        --max_depth=10
        --n_features=50
        --sparsity=0.9
        --batch_size=100000
    
Another example with a full list of parameters:

    python plot_op_tree_ensemble_sparse.py
        --n_trees=100
        --max_depth=10
        --n_features=50
        --batch_size=100000
        --sparsity=0.9
        --tries=3
        --scenario=CUSTOM
        --parallel_tree=80,40
        --parallel_tree_N=128,64
        --parallel_N=50,25
        --batch_size_tree=1,2
        --batch_size_rows=1,2
        --use_node3=0

Another example:

::

    python plot_op_tree_ensemble_sparse.py
        --n_trees=100 --n_features=10 --batch_size=10000 --max_depth=8 -s SHORT        
"""

import logging
import os
import timeit
from typing import Tuple
import numpy
import onnx
from onnx import ModelProto, TensorProto
from onnx.helper import make_graph, make_model, make_tensor_value_info
from pandas import DataFrame, concat
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from onnxruntime import InferenceSession, SessionOptions
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
    optimize_model,
)
from onnx_extended.tools.onnx_nodes import multiply_tree
from onnx_extended.validation.cpu._validation import dense_to_sparse_struct
from onnx_extended.plotting.benchmark import hhistograms
from onnx_extended.args import get_parsed_args
from onnx_extended.ext_test_case import unit_test_going

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
    parallel_tree=("80,160,40", "values to try for parallel_tree"),
    parallel_tree_N=("256,128,64", "values to try for parallel_tree_N"),
    parallel_N=("100,50,25", "values to try for parallel_N"),
    batch_size_tree=("2,4,8", "values to try for batch_size_tree"),
    batch_size_rows=("2,4,8", "values to try for batch_size_rows"),
    use_node3=("0,1", "values to try for use_node3"),
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
        onx = to_onnx(model, X[:1], target_opset={"": 18, "ai.onnx.ml": 3})

        # And wd multiply the trees.
        node = multiply_tree(onx.graph.node[0], n_trees)
        onx = make_model(
            make_graph([node], onx.graph.name, onx.graph.input, onx.graph.output),
            domain=onx.domain,
            opset_imports=onx.opset_import,
            ir_version=onx.ir_version,
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

print(f"batch_size={batch_size}")
print(f"n_features={n_features}")
print(f"n_trees={n_trees}")
print(f"max_depth={max_depth}")
print(f"sparsity={sparsity}")

##############################
# training

filename, Xb, yb = train_model(batch_size, n_features, n_trees, max_depth, sparsity)

print(f"Xb.shape={Xb.shape}")
print(f"yb.shape={yb.shape}")
print(f"measured sparsity={measure_sparsity(Xb)}")

#######################################
# Rewrite the onnx file to use a different kernel
# +++++++++++++++++++++++++++++++++++++++++++++++
#
# The custom kernel is mapped to a custom operator with the same name
# the attributes and domain = `"onnx_extended.ortops.optim.cpu"`.
# We call a function to do that replacement.
# First the current model.

with open(filename, "rb") as f:
    onx = onnx.load(f)
print(onnx_simple_text_plot(onx))

############################
# And then the modified model.


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
        new_op_domain="onnx_extended.ortops.optim.cpu",
        nodes_modes=modes,
        **kwargs,
    )
    if use_sparse:
        del new_onx.graph.input[:]
        new_onx.graph.input.append(
            make_tensor_value_info("X", TensorProto.FLOAT, (None,))
        )
    return new_onx


print("Tranform model to add a custom node.")
onx_modified = transform_model(onx)
print(f"Save into {filename + 'modified.onnx'!r}.")
with open(filename + "modified.onnx", "wb") as f:
    f.write(onx_modified.SerializeToString())
print("done.")
print(onnx_simple_text_plot(onx_modified))

############################
# Same with sparse.


print("Same transformation but with sparse.")
onx_modified_sparse = transform_model(onx, use_sparse=True)
print(f"Save into {filename + 'modified.sparse.onnx'!r}.")
with open(filename + "modified.sparse.onnx", "wb") as f:
    f.write(onx_modified_sparse.SerializeToString())
print("done.")
print(onnx_simple_text_plot(onx_modified_sparse))

#######################################
# Comparing onnxruntime and the custom kernel
# +++++++++++++++++++++++++++++++++++++++++++

print(f"Loading {filename!r}")
sess_ort = InferenceSession(filename, providers=["CPUExecutionProvider"])

r = get_ort_ext_libs()
print(f"Creating SessionOptions with {r!r}")
opts = SessionOptions()
if r is not None:
    opts.register_custom_ops_library(r[0])

print(f"Loading modified {filename!r}")
sess_cus = InferenceSession(
    onx_modified.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)

print(f"Loading modified sparse {filename!r}")
sess_cus_sparse = InferenceSession(
    onx_modified_sparse.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)


print(f"Running once with shape {Xb.shape}.")
base = sess_ort.run(None, {"X": Xb})[0]

print(f"Running modified with shape {Xb.shape}.")
got = sess_cus.run(None, {"X": Xb})[0]
print("done.")

Xb_sp = dense_to_sparse_struct(Xb)
print(f"Running modified sparse with shape {Xb_sp.shape}.")
got_sparse = sess_cus_sparse.run(None, {"X": Xb_sp})[0]
print("done.")

#######################################
# Discrepancies?

diff = numpy.abs(base - got).max()
print(f"Discrepancies: {diff}")

diff = numpy.abs(base - got_sparse).max()
print(f"Discrepancies sparse: {diff}")

########################################
# Simple verification
# +++++++++++++++++++
#
# Baseline with onnxruntime.
t1 = timeit.timeit(lambda: sess_ort.run(None, {"X": Xb}), number=50)
print(f"baseline: {t1}")

#################################
# The custom implementation.
t2 = timeit.timeit(lambda: sess_cus.run(None, {"X": Xb}), number=50)
print(f"new time: {t2}")

#################################
# The custom sparse implementation.
t3 = timeit.timeit(lambda: sess_cus_sparse.run(None, {"X": Xb_sp}), number=50)
print(f"new time sparse: {t3}")

#############################################
# Time for comparison
# +++++++++++++++++++
#
# The custom kernel supports the same attributes as *TreeEnsembleRegressor*
# plus new ones to tune the parallelization. They can be seen in
# `tree_ensemble.cc <https://github.com/sdpython/onnx-extended/
# blob/main/onnx_extended/ortops/optim/cpu/tree_ensemble.cc#L102>`_.
# Let's try out many possibilities.
# The default values are the first ones.

if unit_test_going():
    optim_params = dict(
        parallel_tree=[40],  # default is 80
        parallel_tree_N=[128],  # default is 128
        parallel_N=[50, 25],  # default is 50
        batch_size_tree=[1],  # default is 1
        batch_size_rows=[1],  # default is 1
        use_node3=[0],  # default is 0
    )
elif script_args.scenario in (None, "SHORT"):
    optim_params = dict(
        parallel_tree=[80, 40],  # default is 80
        parallel_tree_N=[128, 64],  # default is 128
        parallel_N=[50, 25],  # default is 50
        batch_size_tree=[1],  # default is 1
        batch_size_rows=[1],  # default is 1
        use_node3=[0],  # default is 0
    )
elif script_args.scenario == "LONG":
    optim_params = dict(
        parallel_tree=[80, 160, 40],
        parallel_tree_N=[256, 128, 64],
        parallel_N=[100, 50, 25],
        batch_size_tree=[1, 2, 4, 8],
        batch_size_rows=[1, 2, 4, 8],
        use_node3=[0, 1],
    )
elif script_args.scenario == "CUSTOM":
    optim_params = dict(
        parallel_tree=list(int(i) for i in script_args.parallel_tree.split(",")),
        parallel_tree_N=list(int(i) for i in script_args.parallel_tree_N.split(",")),
        parallel_N=list(int(i) for i in script_args.parallel_N.split(",")),
        batch_size_tree=list(int(i) for i in script_args.batch_size_tree.split(",")),
        batch_size_rows=list(int(i) for i in script_args.batch_size_rows.split(",")),
        use_node3=list(int(i) for i in script_args.use_node3.split(",")),
    )
else:
    raise ValueError(
        f"Unknown scenario {script_args.scenario!r}, use --help to get them."
    )

cmds = []
for att, value in optim_params.items():
    cmds.append(f"--{att}={','.join(map(str, value))}")
print("Full list of optimization parameters:")
print(" ".join(cmds))

##################################
# Then the optimization for dense


def create_session(onx):
    opts = SessionOptions()
    r = get_ort_ext_libs()
    if r is None:
        raise RuntimeError("No custom implementation available.")
    opts.register_custom_ops_library(r[0])
    return InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )


res = optimize_model(
    onx,
    feeds={"X": Xb},
    transform=transform_model,
    session=create_session,
    baseline=lambda onx: InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    ),
    params=optim_params,
    verbose=True,
    number=script_args.number,
    repeat=script_args.repeat,
    warmup=script_args.warmup,
    sleep=script_args.sleep,
    n_tries=script_args.tries,
)

##################################
# Then the optimization for sparse

res_sparse = optimize_model(
    onx,
    feeds={"X": Xb_sp},
    transform=lambda *args, **kwargs: transform_model(*args, use_sparse=True, **kwargs),
    session=create_session,
    params=optim_params,
    verbose=True,
    number=script_args.number,
    repeat=script_args.repeat,
    warmup=script_args.warmup,
    sleep=script_args.sleep,
    n_tries=script_args.tries,
)


###############################
# And the results.

df_dense = DataFrame(res)
df_dense["input"] = "dense"
df_sparse = DataFrame(res_sparse)
df_sparse["input"] = "sparse"
df = concat([df_dense, df_sparse], axis=0)
df.to_csv("plot_op_tree_ensemble_sparse.csv", index=False)
df.to_excel("plot_op_tree_ensemble_sparse.xlsx", index=False)
print(df.columns)
print(df.head(5))

################################
# Sorting
# +++++++

small_df = df.drop(
    [
        "min_exec",
        "max_exec",
        "repeat",
        "number",
        "context_size",
        "n_exp_name",
    ],
    axis=1,
).sort_values("average")
print(small_df.head(n=10))


################################
# Worst
# +++++

print(small_df.tail(n=10))


#################################
# Plot
# ++++

skeys = ",".join(optim_params.keys())
title = f"TreeEnsemble tuning, n_tries={script_args.tries}\n{skeys}\nlower is better"
ax = hhistograms(df, title=title, keys=("input", "name"))
fig = ax.get_figure()
fig.savefig("plot_op_tree_ensemble_sparse.png")
