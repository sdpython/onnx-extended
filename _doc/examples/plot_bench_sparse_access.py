"""
Evaluating random access for sparse
===================================

Whenever computing the prediction of a tree with a sparse tensor,
is it faster to density first and then to compute the prediction or to
keep the tensor in its sparse representation and do look up?
The parameter *nrnd* can be seen as the depth of a tree.

"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx_extended.ext_test_case import unit_test_going
from onnx_extended.args import get_parsed_args
from onnx_extended.validation.cpu._validation import evaluate_sparse


expose = "repeat,warmup,nrows,ncols,sparsity,nrnd"
script_args = get_parsed_args(
    "plot_bench_sparse_access",
    description=__doc__,
    nrows=(10 if unit_test_going() else 1000, "number of rows"),
    ncols=(10 if unit_test_going() else 100000, "number of columns"),
    sparsity=(
        "0.1,0.2" if unit_test_going() else "0.75,0.8,0.9,0.95,0.99,0.999,0.9999",
        "sparsities to try",
    ),
    repeat=2 if unit_test_going() else 5,
    warmup=2 if unit_test_going() else 2,
    nrnd=(10, "number of random features to access"),
    expose=expose,
)

for att in sorted(expose.split(",")):
    print(f"{att}={getattr(script_args, att)}")

#################################
# Sparse tensor
# +++++++++++++


def make_sparse_random_tensor(n_rows: int, n_cols: int, sparsity: float):
    t = np.random.rand(n_rows, n_cols).astype(np.float32)
    m = np.random.rand(n_rows, n_cols).astype(np.float32)
    t[m <= sparsity] = 0
    return t


sparsity = list(map(float, script_args.sparsity.split(",")))
t = make_sparse_random_tensor(script_args.nrows, script_args.ncols, sparsity[0])
ev = evaluate_sparse(t, script_args.nrnd, script_args.repeat, 3)
print(f"dense:  initialization:{ev[0][0]:1.3g}")
print(f"                access:{ev[0][1]:1.3g}")
print(f"sparse: initialization:{ev[1][0]:1.3g}")
print(f"                access:{ev[1][1]:1.3g}")
print(f"Ratio sparse/dense: {ev[1][1] / ev[0][1]}")

##############################
# If > 1, sparse is slower.

###################################
# Try sparsity
# ++++++++++++
#

data = []
for sp in tqdm(sparsity):
    t = make_sparse_random_tensor(script_args.nrows, script_args.ncols, sp)
    ev = evaluate_sparse(t, script_args.nrnd, script_args.repeat, 3)
    obs = dict(
        dense0=ev[0][0],
        dense1=ev[0][1],
        dense=ev[0][0] + ev[0][1],
        sparse0=ev[1][0],
        sparse1=ev[1][1],
        sparse=ev[1][0] + ev[1][1],
        sparsity=sp,
        rows=t.shape[0],
        cols=t.shape[1],
        repeat=script_args.repeat,
        random=script_args.nrnd,
    )
    data.append(obs)

df = DataFrame(data)
print(df)

############################
# Plots

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[["sparsity", "dense", "sparse"]].set_index("sparsity").plot(
    title="Dense vs Sparsity (access only)", logy=True, ax=ax[0]
)
df[["sparsity", "dense1", "sparse1"]].set_index("sparsity").plot(
    title="Dense vs Sparsity", logy=True, ax=ax[1]
)
fig.tight_layout()
fig.savefig("plot_bench_sparse_access.png")
