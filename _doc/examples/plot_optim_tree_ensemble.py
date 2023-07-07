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

# Training a model
# ++++++++++++++++
"""
import os
import numpy
import onnx
import matplotlib.pyplot as plt
from pandas import DataFrame
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

batch_size = 10000
n_features = 5
n_trees = 10
max_depth = 5

filename = (
    f"plot_optim_tree_ensemble_b{batch_size}-f{n_features}-"
    f"t{n_trees}-d{max_depth}.onnx"
)
if not os.path.exists(filename):
    print(f"Training to get {filename!r}")
    X, y = make_regression(batch_size * 2, n_features=n_features, n_targets=1)
    X, y = X.astype(numpy.float32), y.astype(numpy.float32)
    model = RandomForestRegressor(n_trees, max_depth=max_depth)
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


onx_modified = transform_model(onx)
with open(filename + "modified.onnx", "wb") as f:
    f.write(onx_modified.SerializeToString())
print(onnx_simple_text_plot(onx_modified))

#######################################
# Comparing onnxruntime and the custom kernel
# +++++++++++++++++++++++++++++++++++++++++++

sess_ort = InferenceSession(filename, providers=["CPUExecutionProvider"])

opts = SessionOptions()
r = get_ort_ext_libs()
if r is not None:
    opts.register_custom_ops_library(r[0])

sess_cus = InferenceSession(
    onx_modified.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)

base = sess_ort.run(None, {"X": X[-batch_size:]})[0]
got = sess_cus.run(None, {"X": X[-batch_size:]})[0]

#######################################
# Discrepancies?

diff = numpy.abs(base - got).max()
print(f"Discrepancies: {diff}")


#############################################
# Time for comparison
# +++++++++++++++++++
#
# The custom kernel supports the same attributes as *TreeEnsembleRegressor*
# plus new ones to tune the parallelization. They can be seen in
# `tree_ensemble.cc <https://github.com/sdpython/onnx-extended/
# blob/main/onnx_extended/ortops/optim/cpu/tree_ensemble.cc#L102>`_.
# Let's try out many possibilities.

optim_params = dict(
    parallel_tree=[0, 40, 80],  # default is 80
    parallel_tree_N=[0, 64, 128],  # default is 128
    parallel_N=[0, 25, 50],  # default is 50
    batch_size_tree=[2],  # [2, 4, 8],  # default is 2
    batch_size_rows=[2],  # [2, 4, 8],  # default is 2
    use_node3=[0],  # [0, 1],  # default is 0
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
    number=10,
    repeat=10,
    warmup=10,
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
if dfi.shape[0] > 50:
    dfi = dfi[:50]
dfi = dfi.set_index("short_name")
skeys = ",".join(optim_params.keys())

fig, ax = plt.subplots(1, 1, figsize=(10, dfi.shape[0] / 4))
dfi.plot.barh(title=f"TreeEnsemble tuning\n{skeys}", ax=ax)
b = df.loc[0, "average"]
ax.plot([b, b], [0, df.shape[0]], "r--")
ax.set_xlim(
    [
        (df["min_exec"].min() + df["average"].min()) / 2,
        (df["max_exec"].max() + df["average"].max()) / 2,
    ]
)
fig.tight_layout()
fig.savefig("plot_optim_tree_ensemble.png")
