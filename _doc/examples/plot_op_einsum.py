"""
.. _l-plot-op-einsum:

Compares implementations of Einsum
==================================

This example compares different equations for function :func:`numpy.einsum`.
It compares *numpy* implementation to a custom implementation,
:epkg:`onnxruntime` implementation and :epkg:`opt-einsum` optimisation.
If available, :epkg:`tensorflow` and :epkg:`pytorch` are included as well.
The custom implementation does not do any transpose.
It uses parallelisation and SIMD optimization when the summation
happens on the last axis of both matrices. It only implements
matrix multiplication. We also measure the improvment made with
function :func:`einsum <onnx_extended.tools.einsum.einsum_fct.einsum>`.

Available optimisation
++++++++++++++++++++++

The code shows which optimisation is used for the custom
implementation, *AVX* or *SSE* and the number of available processors,
equal to the default number of used threads to parallelize.
"""

import logging
import numpy
import pandas
import matplotlib.pyplot as plt
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_graph,
    make_node,
    make_tensor_value_info,
    make_opsetid,
)
from onnxruntime import InferenceSession
from onnx_extended.ext_test_case import measure_time, unit_test_going
from tqdm import tqdm
from opt_einsum import contract
from onnx_extended.tools.einsum.einsum_fct import _einsum

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.ticker").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("onnx-extended").setLevel(logging.ERROR)

###################################
# Einsum: common code
# +++++++++++++++++++

try:
    from tensorflow import einsum as tf_einsum, convert_to_tensor
except ImportError:
    tf_einsum = None
try:
    from torch import einsum as torch_einsum, from_numpy
except ImportError:
    torch_einsum = None


def build_ort_einsum(equation, op_version=18):  # opset=13, 14, ...
    onx = make_model(
        make_graph(
            [make_node("Einsum", ["x", "y"], ["z"], equation=equation)],
            equation,
            [
                make_tensor_value_info("x", TensorProto.FLOAT, None),
                make_tensor_value_info("y", TensorProto.FLOAT, None),
            ],
            [make_tensor_value_info("z", TensorProto.FLOAT, None)],
        ),
        opset_imports=[make_opsetid("", op_version)],
        ir_version=9,
    )
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return lambda x, y: sess.run(None, {"x": x, "y": y})


def build_ort_decomposed(equation, op_version=18):  # opset=13, 14, ...
    cache = _einsum(
        equation,
        numpy.float32,
        opset=op_version,
        optimize=True,
        verbose=True,
        runtime="python",
    )
    if not hasattr(cache, "onnx_"):
        cache.build()
    sess = InferenceSession(
        cache.onnx_.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return lambda x, y: sess.run(None, {"X0": x, "X1": y})


def loop_einsum_eq(fct, equation, xs, ys):
    for x, y in zip(xs, ys):
        fct(equation, x, y)


def loop_einsum_eq_th(fct, equation, xs, ys):
    for x, y in zip(xs, ys):
        fct(equation, x, y, nthread=-1)


def loop_einsum(fct, xs, ys):
    for x, y in zip(xs, ys):
        fct(x, y)


def timeit(stmt, ctx, dim, name):
    obs = measure_time(stmt, div_by_number=True, context=ctx, repeat=5, number=1)
    obs["dim"] = dim
    obs["fct"] = name
    return obs


def benchmark_equation(equation):
    # equations
    ort_einsum = build_ort_einsum(equation)
    ort_einsum_decomposed = build_ort_decomposed(equation)
    res = []
    for dim in tqdm([8, 16, 32, 64, 100, 128, 200, 256]):  # , 500, 512]):
        if unit_test_going() and dim > 64:
            break
        xs = [numpy.random.rand(2, dim, 12, 64).astype(numpy.float32) for _ in range(5)]
        ys = [numpy.random.rand(2, dim, 12, 64).astype(numpy.float32) for _ in range(5)]

        # numpy
        ctx = dict(
            equation=equation,
            xs=xs,
            ys=ys,
            einsum=numpy.einsum,
            loop_einsum=loop_einsum,
            loop_einsum_eq=loop_einsum_eq,
            loop_einsum_eq_th=loop_einsum_eq_th,
        )
        obs = timeit(
            "loop_einsum_eq(einsum, equation, xs, ys)", ctx, dim, "numpy.einsum"
        )
        res.append(obs)

        # opt-einsum
        ctx["einsum"] = contract
        obs = timeit("loop_einsum_eq(einsum, equation, xs, ys)", ctx, dim, "opt-einsum")
        res.append(obs)

        # onnxruntime
        ctx["einsum"] = ort_einsum
        obs = timeit("loop_einsum(einsum, xs, ys)", ctx, dim, "ort-einsum")
        res.append(obs)

        # onnxruntime decomposed
        ctx["einsum"] = ort_einsum_decomposed
        obs = timeit("loop_einsum(einsum, xs, ys)", ctx, dim, "ort-dec")
        res.append(obs)

        if tf_einsum is not None:
            # tensorflow
            ctx["einsum"] = tf_einsum
            ctx["xs"] = [convert_to_tensor(x) for x in xs]
            ctx["ys"] = [convert_to_tensor(y) for y in ys]
            obs = timeit(
                "loop_einsum_eq(einsum, equation, xs, ys)", ctx, dim, "tf-einsum"
            )
            res.append(obs)

        if torch_einsum is not None:
            # torch
            ctx["einsum"] = torch_einsum
            ctx["xs"] = [from_numpy(x) for x in xs]
            ctx["ys"] = [from_numpy(y) for y in ys]
            obs = timeit(
                "loop_einsum_eq(einsum, equation, xs, ys)", ctx, dim, "torch-einsum"
            )
            res.append(obs)

    # Dataframes
    df = pandas.DataFrame(res)
    piv = df.pivot(index="dim", columns="fct", values="average")

    rs = piv.copy()
    for c in ["ort-einsum", "ort-dec", "tf-einsum", "torch-einsum", "opt-einsum"]:
        if c not in rs.columns:
            continue
        rs[c] = rs["numpy.einsum"] / rs[c]
    rs["numpy.einsum"] = 1.0

    # Graphs.
    _fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    piv.plot(
        logx=True,
        logy=True,
        ax=ax[0],
        title=f"Einsum benchmark\n{equation} -- (2, N, 12, 64) lower better",
    )
    ax[0].legend(prop={"size": 9})
    rs.plot(
        logx=True,
        logy=True,
        ax=ax[1],
        title="Einsum Speedup, baseline=numpy\n%s -- (2, N, 12, 64)"
        " higher better" % equation,
    )
    ax[1].plot([min(rs.index), max(rs.index)], [0.5, 0.5], "g--")
    ax[1].plot([min(rs.index), max(rs.index)], [2.0, 2.0], "g--")
    ax[1].legend(prop={"size": 9})

    return df, rs, ax


###################################
# First equation: bsnh,btnh->bnts
# +++++++++++++++++++++++++++++++
#
# The decomposition of this equation without einsum function gives
# the following.
#
#  .. gdot::
#       :script:
#
#       from onnx_extended.tools.einsum import decompose_einsum_equation
#       dec = decompose_einsum_equation(
#           'bsnh,btnh->bnts', strategy='numpy', clean=True)
#       print(dec.to_dot())

dfs = []
equation = "bsnh,btnh->bnts"
df, piv, ax = benchmark_equation(equation)
df.pivot(index="fct", columns="dim", values="average")
dfs.append(df)

###################################
# Second equation: bshn,bthn->bnts
# ++++++++++++++++++++++++++++++++
#
# The summation does not happen on the last axis but
# on the previous one.
# Is it worth transposing before doing the summation...
# The decomposition of this equation without einsum function gives
# the following.
#
#  .. gdot::
#       :script:
#
#       from onnx_extended.tools.einsum import decompose_einsum_equation
#       dec = decompose_einsum_equation(
#           'bshn,bthn->bnts', strategy='numpy', clean=True)
#       print(dec.to_dot())

equation = "bshn,bthn->bnts"
df, piv, ax = benchmark_equation(equation)
df.pivot(index="fct", columns="dim", values="average")
dfs.append(df)

###################################
# Third equation: bhsn,bhtn->bnts
# +++++++++++++++++++++++++++++++
#
# The summation does not happen on the last axis but
# on the second one. It is worth transposing before multiplying.
# The decomposition of this equation without einsum function gives
# the following.
#
#  .. gdot::
#       :script:
#
#       from onnx_extended.tools.einsum import decompose_einsum_equation
#       dec = decompose_einsum_equation(
#           'bhsn,bhtn->bnts', strategy='numpy', clean=True)
#       print(dec.to_dot())

equation = "bhsn,bhtn->bnts"
df, piv, ax = benchmark_equation(equation)
df.pivot(index="fct", columns="dim", values="average")
dfs.append(df)

####################################
# Conclusion
# ++++++++++
#
# pytorch seems quite efficient on these examples.
# The custom implementation was a way to investigate
# the implementation of einsum and find some ways to optimize it.

merged = pandas.concat(dfs)
name = "einsum"
merged.to_csv(f"plot_{name}.csv", index=False)
merged.to_excel(f"plot_{name}.xlsx", index=False)
plt.savefig(f"plot_{name}.png")

# plt.show()
