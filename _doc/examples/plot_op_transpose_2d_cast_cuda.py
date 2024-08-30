"""
.. _l-example-op-transpose2dcast_cuda:

==============================
Fuse Tranpose and Cast on CUDA
==============================

This configuration happens in a :epkg:`Llama` model.

::

    output = Cast(Transpose(X), to=FLOAT16)

Where the shapes are:

* X: 4096,4096

Transpose + Cast
================
"""

from onnx_extended.args import get_parsed_args

script_args = get_parsed_args(
    "plot_op_transpose_2d_cast",
    description=__doc__,
    config=(
        "small",
        "small, short optimization (default), "
        "medium for medium sizes, "
        "large for big sizes",
        "llama for a specific case on llama",
    ),
    warmup=3,
    repeat=5,
    itype=(10, "1 or 10 for float or float16"),
    expose="config,itype,warmup,repeat",
)

import time
import numpy as np
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from tqdm import tqdm
import onnx.helper as oh
from onnx import TensorProto
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

itype = script_args.itype
dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
config = script_args.config
print(f"config={config}")
print(f"itype={itype}, dtype={dtype}")

if config == "small":
    sizes = (256, 512, 1024)
elif config == "medium":
    sizes = (512, 1024, 2048)
elif config == "large":
    sizes = (1024, 2048, 4096, 8192)
elif config == "llama":
    sizes = (2048, 4096, 8192)
else:
    try:
        sizes = list(map(int, config.split(",")))
    except (ValueError, TypeError) as e:
        raise AssertionError(f"Unexpected config value {config!r}.") from e


def get_model(fused=False, itype=TensorProto.FLOAT):
    iitype = TensorProto.FLOAT if itype == TensorProto.FLOAT16 else TensorProto.FLOAT16
    suffix = "32" if itype == TensorProto.FLOAT else "16"
    if fused:
        nodes = [
            oh.make_node(
                f"Transpose2DCastFP{suffix}",
                ["X"],
                ["Y"],
                domain="onnx_extended.ortops.optim.cuda",
            )
        ]
    else:
        nodes = [
            oh.make_node("Transpose", ["X"], ["xt"], perm=[1, 0]),
            oh.make_node("Cast", ["xt"], ["Y"], to=itype),
        ]
    model = oh.make_model(
        oh.make_graph(
            nodes,
            "g",
            [oh.make_tensor_value_info("X", iitype, ["a", "b"])],
            [oh.make_tensor_value_info("Y", itype, ["b", "a"])],
        ),
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ],
        ir_version=9,
    )
    return model


model = get_model(itype=itype)
print(onnx_simple_text_plot(model))

###################################
# Models
# ======


def get_session(model):
    import onnxruntime
    from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return None

    opts = onnxruntime.SessionOptions()
    opts.register_custom_ops_library(get_ort_ext_libs()[0])
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(),
        opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return sess


X = np.random.randn(64, 32).astype(
    np.float16 if itype == TensorProto.FLOAT else np.float32
)
feeds = dict(X=X)

sess1 = get_session(model)
if sess1 is not None:
    for k, v in feeds.items():
        print(k, v.dtype, v.shape)
    expected = sess1.run(None, feeds)[0]
    print(expected[:4, :4])

##################################################
# Same model but using the fused op.

model = get_model(fused=True, itype=itype)
print(onnx_simple_text_plot(model))

sess2 = get_session(model)
if sess2 is not None:
    got = sess2.run(None, feeds)[0]
    print(got[:4, :4])
    assert_almost_equal(expected, got)

#################################################
# Benchmark
# =========


def move_inputs(sess, feeds):
    from onnxruntime.capi._pybind_state import (
        SessionIOBinding,
        OrtDevice as C_OrtDevice,
        OrtValue as C_OrtValue,
    )

    input_names = [i.name for i in sess.get_inputs()]

    ort_device = C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)

    feed_ort_value = [
        (name, C_OrtValue.ortvalue_from_numpy(feeds[name], ort_device))
        for name in input_names
    ]

    bind = SessionIOBinding(sess._sess)
    for name, value in feed_ort_value:
        bind.bind_input(
            name, ort_device, feeds[name].dtype, value.shape(), value.data_ptr()
        )
    for o in sess.get_outputs():
        bind.bind_output(o.name, ort_device)
    return bind, feed_ort_value


def benchmark(
    sess, sizes, config, label, itype, times_col: int = 1, times_indices: int = 1
):

    data = []
    for size in tqdm(sizes):

        X = np.random.randn(size, size).astype(
            np.float16 if itype == TensorProto.FLOAT else np.float32
        )
        feeds = dict(X=X)
        bind, cuda_feeds = move_inputs(sess, feeds)

        begin = time.perf_counter()
        for _i in range(script_args.warmup):
            # sess.run(None, feeds)
            sess._sess.run_with_iobinding(bind, None)
        warmup = time.perf_counter() - begin

        times = []
        for _i in range(script_args.repeat):
            begin = time.perf_counter()
            # sess.run(None, feeds)
            sess._sess.run_with_iobinding(bind, None)
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


if sess1 is not None:

    print(f"sizes={sizes}")

    data_nd1 = benchmark(sess1, sizes, script_args.config, "Not Fused", itype=itype)

#######################################
# Fused.

if sess2 is not None:

    data_nd2 = benchmark(sess2, sizes, script_args.config, "Fused", itype=itype)


##########################################
# Data
# ++++

if sess2 is not None:

    df = DataFrame(data_nd1 + data_nd2)
    df.to_csv("plot_op_transpose_2d_cast_cuda.csv", index=False)
    df.to_csv("plot_op_transpose_2d_cast_cuda.xlsx", index=False)
    print(df.head())

#####################
# Pivot.

if sess2 is not None:

    pivot = df.pivot(index="size", columns="label", values="time")
    pivot["ratio"] = pivot["Not Fused"] / pivot["Fused"]
    print(pivot)

    ax = pivot[["Not Fused", "Fused"]].plot(
        logx=True,
        logy=True,
        title=(
            f"Not Fused/Fused implementation for Transpose + "
            f"Cast on CUDA\nitype={itype}"
        ),
    )
    ax.get_figure().savefig("plot_op_transpose_2d_cast_cuda.png")

##############################
# It seems worth it to combine both operators.
