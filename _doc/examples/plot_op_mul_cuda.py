"""
.. _l-example-op-mul_cuda:

Fusing multiplication operators on CUDA
=======================================

The examples compare the performaance of two fused operators Mul
with the unfused sequence.

Cache Performance
+++++++++++++++++
"""

from onnx_extended.args import get_parsed_args

script_args = get_parsed_args(
    "plot_op_mul_cuda",
    description=__doc__,
    config=(
        "small",
        "small, short optimization (default), "
        "medium for medium sizes, "
        "large for big sizes",
    ),
    warmup=3,
    repeat=5,
    itype=(1, "1 or 10 for float or float16"),
    expose="config,itype,warmup,repeat",
)

itype = script_args.itype
config = script_args.config
print(f"config={config}")
print(f"itype={itype}")

if config == "small":
    sizes = (8, 16)
elif config == "medium":
    sizes = (256, 512, 1024)
elif config == "large":
    sizes = (1024, 2048, 4096, 8192)
else:
    try:
        sizes = list(map(int, config.split(",")))
    except (ValueError, TypeError) as e:
        raise AssertionError(f"Unexpected config value {config!r}.") from e

import time
import numpy as np
import onnx.helper as oh
from tqdm import tqdm
from pandas import DataFrame
from onnxruntime import InferenceSession, SessionOptions
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_extended.ortops.optim.cuda import get_ort_ext_libs


def get_model1(itype):
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Mul", ["X", "Y"], ["xy"]),
                oh.make_node("Mul", ["xy", "Z"], ["xyz"]),
                oh.make_node("Mul", ["Y", "X"], ["yx"]),
                oh.make_node("Mul", ["xyz", "yx"], ["final"]),
            ],
            "nd",
            [
                oh.make_tensor_value_info("X", itype, [None, None]),
                oh.make_tensor_value_info("Y", itype, [None, None]),
                oh.make_tensor_value_info("Z", itype, [None, None]),
            ],
            [oh.make_tensor_value_info("final", itype, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )


print(onnx_simple_text_plot(get_model1(itype)))


########################################
# And the other model


def get_model2(itype):
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "MulMul",
                    ["X", "Y", "Z"],
                    ["xyz"],
                    domain="onnx_extented.ortops.optim.cuda",
                ),
                oh.make_node(
                    "MulMul",
                    ["Y", "X", "xyz"],
                    ["final"],
                    domain="onnx_extented.ortops.optim.cuda",
                ),
            ],
            "nd",
            [
                oh.make_tensor_value_info("X", itype, [None, None]),
                oh.make_tensor_value_info("Y", itype, [None, None]),
                oh.make_tensor_value_info("Z", itype, [None, None]),
            ],
            [oh.make_tensor_value_info("final", itype, [None, None])],
        ),
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("onnx_extented.ortops.optim.cuda", 1),
        ],
        ir_version=9,
    )


print(onnx_simple_text_plot(get_model2(itype)))

###########################################
# InferenceSession
# ++++++++++++++++

dtype = np.float32 if itype == 1 else np.float16

x = np.random.randn(16, 16).astype(dtype)
y = np.random.randn(16, 16).astype(dtype)
z = np.random.randn(16, 16).astype(dtype)
feeds = dict(X=x, Y=y, Z=z)

sess1 = InferenceSession(
    get_model1(itype).SerializeToString(), providers=["CUDAExecutionProvider"]
)
expected = sess1.run(None, feeds)[0]

#########################################
# The other model.

opts = SessionOptions()
opts.register_custom_ops_library(get_ort_ext_libs()[0])

sess2 = InferenceSession(
    get_model2(itype).SerializeToString(), opts, providers=["CUDAExecutionProvider"]
)
got = sess2.run(None, feeds)[0]

########################################
# Discrepancies

diff = np.abs(got - expected).max()
print(f"diff={diff}")


############################################
# Benchmark
# +++++++++
#
# Forst model.


def benchmark(sess, sizes, label):

    data = []
    for size in tqdm(sizes):

        x = np.random.randn(size, size).astype(dtype)
        y = np.random.randn(size, size).astype(dtype)
        z = np.random.randn(size, size).astype(dtype)
        feeds = dict(X=x, Y=y, Z=z)

        begin = time.perf_counter()
        for i in range(script_args.warmup):
            sess.run(None, feeds)
        warmup = time.perf_counter() - begin

        times = []
        for i in range(script_args.repeat):
            begin = time.perf_counter()
            sess.run(None, feeds)
            times.append(time.perf_counter() - begin)

        npt = np.array(times)
        obs = dict(
            warmup=warmup,
            time=npt.mean(),
            std=npt.std(),
            min=npt.min(),
            max=npt.max(),
            repeat=script_args.repeat,
            size=size,
            label=label,
        )
        data.append(obs)
    return data


#######################################
# Not Fused.

print(f"sizes={sizes}")

data_mul = benchmark(sess1, sizes, "Not Fused")

#######################################
# Fused.

data_mulmul = benchmark(sess2, sizes, "Fused")


##########################################
# Data
# ++++

df = DataFrame(data_mul + data_mulmul)
df.to_csv("plot_op_mul_cuda.csv", index=False)
df.to_csv("plot_op_mul_cuda.xlsx", index=False)
print(df.head())

#####################
# Pivot.

pivot = df.pivot(index="size", columns="label", values="time")
print(pivot)


std = df.pivot(index="size", columns="label", values="std")
ax = pivot.plot(
    logx=True,
    logy=True,
    title=f"Fused/Unfused element wise multiplication on CUDA\nitype={itype}",
)
ax.get_figure().savefig("plot_op_mul_cuda.png")
