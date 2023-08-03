"""
.. _l-plot-optim-tree-ensemble:

TreeEnsemble optimization
=========================

The execution of a TreeEnsembleRegressor can lead to very different results
depending on how the computation is parallelized. By trees,
by rows, by both, for only one row, for a short batch of rows, a longer one.
The implementation in :epkg:`onnxruntime` does not let the user changed
the predetermined settings but a custom kernel might. That's what this example
is measuring.

The default set of optimized parameters is very short and is meant to be executed
fast. Many more parameters can be tried.

::

    python plot_optim_tree_ensemble --scenario=LONG

To change the training parameters:

::

    python plot_optim_tree_ensemble.py
        --n_trees=100
        --max_depth=10
        --n_features=50
        --batch_size=100000 
"""
import os
import timeit
import numpy
import onnx
from onnx.reference import ReferenceEvaluator
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from onnxruntime import InferenceSession, SessionOptions
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
    optimize_model,
)
from onnx_extended.ext_test_case import get_parsed_args, unit_test_going

script_args = get_parsed_args(
    "plot_optim_tree_ensemble",
    description=__doc__,
    scenarios={
        "SHORT": "short optimization (default)",
        "LONG": "test more options",
        "CUSTOM": "use values specified by the command line",
    },
    n_features=(2 if unit_test_going() else 5, "number of features to generate"),
    n_trees=(3 if unit_test_going() else 10, "number of trees to train"),
    max_depth=(2 if unit_test_going() else 5, "max_depth"),
    batch_size=(1000 if unit_test_going() else 10000, "batch size"),
    parallel_tree=("80,160,40", "values to try for parallel_tree"),
    parallel_tree_N=("256,128,64", "values to try for parallel_tree_N"),
    parallel_N=("100,50,25", "values to try for parallel_N"),
    batch_size_tree=("2,4,8", "values to try for batch_size_tree"),
    batch_size_rows=("2,4,8", "values to try for batch_size_rows"),
    use_node3=("0,1", "values to try for use_node3"),
    expose="",
)


################################
# Training a model
# ++++++++++++++++

batch_size = script_args.batch_size
n_features = script_args.n_features
n_trees = script_args.n_trees
max_depth = script_args.max_depth

filename = (
    f"plot_optim_tree_ensemble_b{batch_size}-f{n_features}-"
    f"t{n_trees}-d{max_depth}.onnx"
)
if not os.path.exists(filename):
    print(f"Training to get {filename!r}")
    X, y = make_regression(batch_size * 2, n_features=n_features, n_targets=1)
    X, y = X.astype(numpy.float32), y.astype(numpy.float32)
    model = RandomForestRegressor(n_trees, max_depth=max_depth, verbose=2)
    model.fit(X[:batch_size], y[:batch_size])
    onx = to_onnx(model, X[:1])
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
else:
    X, y = make_regression(batch_size, n_features=n_features, n_targets=1)
    X, y = X.astype(numpy.float32), y.astype(numpy.float32)


#######################################
# Rewrite the onnx file to use a different kernel
# +++++++++++++++++++++++++++++++++++++++++++++++
#
# The custom kernel is mapped to a custom operator with the same name
# the attributes and domain = `"onnx_extented.ortops.optim.cpu"`.
# We call a function to do that replacement.
# First the current model.

with open(filename, "rb") as f:
    onx = onnx.load(f)
print(onnx_simple_text_plot(onx))

############################
# And then the modified model.


def transform_model(onx, **kwargs):
    att = get_node_attribute(onx.graph.node[0], "nodes_modes")
    modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))
    return change_onnx_operator_domain(
        onx,
        op_type="TreeEnsembleRegressor",
        op_domain="ai.onnx.ml",
        new_op_domain="onnx_extented.ortops.optim.cpu",
        nodes_modes=modes,
        **kwargs,
    )


print("Tranform model to add a custom node.")
onx_modified = transform_model(onx)
print(f"Save into {filename + 'modified.onnx'!r}.")
with open(filename + "modified.onnx", "wb") as f:
    f.write(onx_modified.SerializeToString())
print("done.")
print(onnx_simple_text_plot(onx_modified))

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

print("Loading modified {filename!r}")
sess_cus = InferenceSession(
    onx_modified.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)

print(f"Running once with shape {X[-batch_size:].shape}.")
base = sess_ort.run(None, {"X": X[-batch_size:]})[0]
print(f"Running modified with shape {X[-batch_size:].shape}.")
got = sess_cus.run(None, {"X": X[-batch_size:]})[0]
print("done.")

#######################################
# Discrepancies?

diff = numpy.abs(base - got).max()
print(f"Discrepancies: {diff}")

########################################
# Simple verification
# +++++++++++++++++++
#
# Baseline with onnxruntime.
t1 = timeit.timeit(lambda: sess_ort.run(None, {"X": X[-batch_size:]}), number=50)
print(f"baseline: {t1}")

#################################
# The custom implementation.
t2 = timeit.timeit(lambda: sess_cus.run(None, {"X": X[-batch_size:]}), number=50)
print(f"new time: {t2}")

#################################
# The same implementation but ran from the onnx python backend.
ref = CReferenceEvaluator(filename)
ref.run(None, {"X": X[-batch_size:]})
t3 = timeit.timeit(lambda: ref.run(None, {"X": X[-batch_size:]}), number=50)
print(f"CReferenceEvaluator: {t3}")

#################################
# The python implementation but from the onnx python backend.
ref = ReferenceEvaluator(filename)
ref.run(None, {"X": X[-batch_size:]})
t4 = timeit.timeit(lambda: ref.run(None, {"X": X[-batch_size:]}), number=5)
print(f"ReferenceEvaluator: {t4} (only 5 times instead of 50)")


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
        batch_size_tree=[2],  # default is 2
        batch_size_rows=[2],  # default is 2
        use_node3=[0],  # default is 0
    )
elif script_args.scenario in (None, "SHORT"):
    optim_params = dict(
        parallel_tree=[80, 40],  # default is 80
        parallel_tree_N=[128, 64],  # default is 128
        parallel_N=[50, 25],  # default is 50
        batch_size_tree=[2],  # default is 2
        batch_size_rows=[2],  # default is 2
        use_node3=[0],  # default is 0
    )
elif script_args.scenario == "LONG":
    optim_params = dict(
        parallel_tree=[80, 160, 40],
        parallel_tree_N=[256, 128, 64],
        parallel_N=[100, 50, 25],
        batch_size_tree=[2, 4, 8],
        batch_size_rows=[2, 4, 8],
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

##################################
# Then the optimization.


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
    feeds={"X": X[-batch_size:]},
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

###############################
# And the results.

df = DataFrame(res)
df.to_csv("plot_optim_tree_ensemble.csv", index=False)
df.to_excel("plot_optim_tree_ensemble.xlsx", index=False)
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

dfi = df[["short_name", "average"]].sort_values("average").reset_index(drop=True)
baseline = dfi[dfi.short_name.str.contains("baseline")]
not_baseline = dfi[~dfi.short_name.str.contains("baseline")].reset_index(drop=True)
if not_baseline.shape[0] > 50:
    not_baseline = not_baseline[:50]
merged = concat([baseline, not_baseline], axis=0)
merged = merged.sort_values("average").reset_index(drop=True).set_index("short_name")
skeys = ",".join(optim_params.keys())

fig, ax = plt.subplots(1, 1, figsize=(10, merged.shape[0] / 4))
merged.plot.barh(
    ax=ax, title=f"TreeEnsemble tuning, n_tries={script_args.tries}\n{skeys}"
)
b = df.loc[0, "average"]
ax.plot([b, b], [0, df.shape[0]], "r--")
ax.set_xlim(
    [
        (df["min_exec"].min() + df["average"].min()) / 2,
        (df["max_exec"].max() + df["average"].max()) / 2,
    ]
)
ax.set_xscale("log")

fig.tight_layout()
fig.savefig("plot_optim_tree_ensemble.png")
