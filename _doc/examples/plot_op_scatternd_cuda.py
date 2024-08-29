"""
.. _l-example-op-scatternd_cuda:

=====================================
Optimizing ScatterND operator on CUDA
=====================================

How to parallelize something like the following?

ScatterND
=========

This configuration happens in a :epkg:`Llama` model.

::

    gradient = ScatterND(zeros, indices, updates)

Where the shapes are:

* zeros: 32000x4096
* indices: 2x1024x1
* updates: 2x1024x4096
"""

from onnx_extended.args import get_parsed_args

script_args = get_parsed_args(
    "plot_op_scatternd_cuda",
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
    itype=(1, "1 or 10 for float or float16"),
    expose="config,itype,warmup,repeat",
)

import time
import numpy as np
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from tqdm import tqdm
import onnx.helper as oh
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
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
    sizes = (16000, 32000)
else:
    try:
        sizes = list(map(int, config.split(",")))
    except (ValueError, TypeError) as e:
        raise AssertionError(f"Unexpected config value {config!r}.") from e


def get_model(d3=True, optimize=False, shape_input=False, itype=TensorProto.FLOAT):
    indices_shape = ["i", "j", 1] if d3 else ["m", 1]
    updates_shape = ["i", "j", "b"] if d3 else ["m", "b"]
    kwargs = dict(reduction="add")
    if shape_input:
        kwargs["domain"] = "onnx_extended.ortops.optim.cuda"
    if optimize:
        kwargs["strategy"] = "optimize"

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "ScatterNDOfShape" if shape_input else "ScatterND",
                    ["shape" if shape_input else "X", "indices", "updates"],
                    ["Y"],
                    **kwargs,
                )
            ],
            "g",
            [
                (
                    oh.make_tensor_value_info("shape", TensorProto.INT64, ["s"])
                    if shape_input
                    else oh.make_tensor_value_info("X", itype, ["a", "b"])
                ),
                oh.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                oh.make_tensor_value_info("updates", itype, updates_shape),
            ],
            [oh.make_tensor_value_info("Y", itype, ["a", "b"])],
        ),
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ],
        ir_version=9,
    )
    return model


model = get_model()
print(onnx_simple_text_plot(model))


##########################################
# Let's see the evaluation by the ReferenceEvaluator.


def _scatter_nd_impl(data, indices, updates, reduction=None, verbose=False):  # type: ignore
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if verbose:
            print(f"updates for i={i}, indices={indices[i]}, updates={updates[i]}")
        if reduction == "add":
            output[tuple(indices[i])] += updates[i]
        elif reduction == "mul":
            output[tuple(indices[i])] *= updates[i]
        elif reduction == "max":
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None, optimize=None):  # type: ignore
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction, verbose=True)
        return (y,)


class ScatterNDOfShape(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, shape, indices, updates, reduction=None, optimize=None):  # type: ignore
        data = np.zeros(tuple(shape.tolist()), dtype=updates.dtype)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


shape = (5, 7)
X = np.zeros(shape, dtype=dtype)
indices = np.zeros((2, 10, 1)).astype(np.int64)
indices[:, ::2, 0] = 3
updates = np.ones((2, 10, 7)).astype(dtype)
feeds = {"X": X, "indices": indices, "updates": updates}


ref = ReferenceEvaluator(model, new_ops=[ScatterND])
got = ref.run(None, feeds)[0]
print(got)


###########################################
# To generalize, let's change the shapes.

model = get_model(d3=False, itype=itype)
print(onnx_simple_text_plot(model))


new_indices = indices.reshape((-1, 1))
new_updates = updates.reshape((-1, updates.shape[-1]))
feeds = {"X": X, "indices": indices, "updates": updates}

ref = ReferenceEvaluator(model, new_ops=[ScatterND])
got = ref.run(None, feeds)[0]
print(got)


##############################################
# First scenario
# ==============

model = get_model(d3=False, shape_input=True, itype=itype)
print(onnx_simple_text_plot(model))


feeds = {
    "shape": np.array(X.shape, dtype=np.int64),
    "indices": indices.reshape((-1, 1)),
    "updates": updates.reshape((-1, updates.shape[-1])),
}

ref = ReferenceEvaluator(model, new_ops=[ScatterNDOfShape])
expected = ref.run(None, feeds)[0]
print(expected)


###################################
# With onnxruntime


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


sess1 = get_session(model)
if sess1 is not None:
    for k, v in feeds.items():
        print(k, v.dtype, v.shape)
    got = sess1.run(None, feeds)[0]
    print(got)
    assert_almost_equal(expected, got)

##################################################
# Same model but using an optimization to compute it.

model = get_model(d3=False, shape_input=True, optimize=True, itype=itype)
print(onnx_simple_text_plot(model))

sess2 = get_session(model)
if sess2 is not None:
    got = sess2.run(None, feeds)[0]
    print(got)
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

        if config == "llama":
            # zeros: 32000x4096
            # indices: 2x1024x1
            # updates: 2x1024x4096
            nrow, ncol = size, 4096
            nind = 1024
        else:
            nrow, ncol = size, int(size * times_col)
            nind = int(size * times_indices)

        shape = np.array([nrow, ncol], dtype=np.int64)
        indices = np.array(
            [np.random.randint(0, nrow - 1) for _ in range(nind)], dtype=np.int64
        ).reshape((-1, 1))
        updates = np.random.randn(nind, ncol).astype(
            np.float32 if itype == TensorProto.FLOAT else np.float16
        )
        feeds = dict(shape=shape, indices=indices, updates=updates)
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

    data_nd1 = benchmark(
        sess1, sizes, script_args.config, "Atomic/Not Fused", itype=itype
    )

#######################################
# Fused.

if sess2 is not None:

    data_nd2 = benchmark(
        sess2, sizes, script_args.config, "No Atomic/Fused", itype=itype
    )


##########################################
# Data
# ++++

if sess2 is not None:

    df = DataFrame(data_nd1 + data_nd2)
    df.to_csv("plot_op_scatternd_cuda.csv", index=False)
    df.to_csv("plot_op_scatternd_cuda.xlsx", index=False)
    print(df.head())

#####################
# Pivot.

if sess2 is not None:

    pivot = df.pivot(index="size", columns="label", values="time")
    pivot["ratio"] = pivot["Atomic/Not Fused"] / pivot["No Atomic/Fused"]
    print("Speed up compare to the onnx standaed.")
    print(pivot)

    ax = pivot[["Atomic/Not Fused", "No Atomic/Fused"]].plot(
        logx=True,
        logy=True,
        title=f"Atomic/No-Atomic implementation for ScatterND on CUDA\nitype={itype}",
    )
    ax.get_figure().savefig("plot_op_scatternd_cuda.png")

##############################
# The best choice depends on the input sizes,
# For big matrices, the use of atomic is slowing down
# the computation.
