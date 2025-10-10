"""
.. _l-example-op-scatternd_mask_cuda:

============================================
Optimizing Masked ScatterND operator on CUDA
============================================

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
    "plot_op_scatternd_mask_cuda",
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
import onnx.numpy_helper as onh
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
    sizes = (1024, 2048, 4096, 6144, 8192)
elif config == "llama":
    sizes = (16000, 32000)
else:
    try:
        sizes = list(map(int, config.split(",")))
    except (ValueError, TypeError) as e:
        raise AssertionError(f"Unexpected config value {config!r}.") from e


def get_model(op_type="ScatterND", itype=TensorProto.FLOAT):
    indices_shape = ["i", "j", 1]
    updates_shape = ["i", "j", "b"]
    dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
    if op_type == "ScatterND":
        nodes = [
            oh.make_node(
                "ConstantOfShape",
                ["shape"],
                ["data"],
                value=onh.from_array(np.array([0], dtype=dtype)),
            ),
            oh.make_node(
                "Constant",
                [],
                ["mone"],
                value=onh.from_array(np.array([-1], dtype=np.int64)),
            ),
            oh.make_node("Equal", ["indices", "mone"], ["eq"]),
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0], dtype=dtype)),
            ),
            oh.make_node("Where", ["eq", "zero", "updates"], ["new_updates"]),
            oh.make_node(
                op_type, ["data", "indices", "new_updates"], ["Y"], reduction="add"
            ),
        ]
    elif op_type == "ScatterNDOfShape":
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["mone"],
                value=onh.from_array(np.array([-1], dtype=np.int64)),
            ),
            oh.make_node("Equal", ["indices", "mone"], ["eq"]),
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0], dtype=dtype)),
            ),
            oh.make_node("Where", ["eq", "zero", "updates"], ["new_updates"]),
            oh.make_node(
                op_type,
                ["shape", "indices", "new_updates"],
                ["Y"],
                strategy="optimize",
                reduction="add",
                domain="onnx_extended.ortops.optim.cuda",
            ),
        ]
    elif op_type == "MaskedScatterNDOfShape":
        nodes = [
            oh.make_node(
                op_type,
                ["shape", "indices", "updates"],
                ["Y"],
                maskedValue=-1,
                reduction="add",
                domain="onnx_extended.ortops.optim.cuda",
            ),
        ]
    else:
        raise ValueError(f"Unkown value for op_type={op_type!r}.")

    model = oh.make_model(
        oh.make_graph(
            nodes,
            "g",
            [
                oh.make_tensor_value_info("shape", TensorProto.INT64, ["s"]),
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
# Let's see the evaluation by the ReferenceEvaluator for the three
# proposed models.


def _scatter_nd_impl(data, indices, updates, reduction=None, verbose=False):  # type: ignore
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if verbose:
            print(f"updates for i={i}, indices={indices[i]}, updates={updates[i]}")
        assert reduction == "add", f"not implemented for reduction={reduction!r}"
        output[tuple(indices[i])] += updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None, optimize=None):  # type: ignore
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction, verbose=True)
        return (y,)


class ScatterNDOfShape(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, shape, indices, updates, reduction=None, strategy=None):  # type: ignore
        data = np.zeros(tuple(shape.tolist()), dtype=updates.dtype)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


class MaskedScatterNDOfShape(OpRun):
    op_domain = "onnx_extended.ortops.optim.cuda"

    def _run(self, shape, indices, updates, reduction=None, maskedValue=None):
        data = np.zeros(shape, dtype=updates.dtype)
        new_updates = np.where(indices == maskedValue, 0, updates)
        y = _scatter_nd_impl(data, indices, new_updates, reduction=reduction)
        return (y,)


shape = np.array([5, 7], dtype=np.int64)
indices = np.ones((2, 10, 1)).astype(np.int64)
indices[0, ::2, 0] = 3
indices[1, ::2, 0] = 1
indices[:, 1::4, 0] = -1
updates = np.ones((2, 10, 7)).astype(dtype)
feeds = {"shape": shape, "indices": indices, "updates": updates}
baseline = None

for op_type in ["ScatterND", "ScatterNDOfShape", "MaskedScatterNDOfShape"]:
    print("-----------------------------------------------")
    print(f"op_type={op_type}")
    model = get_model(op_type)
    print(onnx_simple_text_plot(model))

    ref = ReferenceEvaluator(
        model, new_ops=[ScatterND, ScatterNDOfShape, MaskedScatterNDOfShape]
    )
    got = ref.run(None, feeds)[0]
    print(got)

    if baseline is None:
        baseline = got
    assert_almost_equal(baseline, got)


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


for op_type in ["ScatterND", "ScatterNDOfShape", "MaskedScatterNDOfShape"]:
    print("-----------------------------------------------")
    print(f"op_type={op_type}")
    model = get_model(op_type)
    print(onnx_simple_text_plot(model))

    sess = get_session(model)
    if sess is not None:
        got = ref.run(None, feeds)[0]
        print(got)
    else:
        print("onnxruntime is not available.")

    if baseline is None:
        baseline = got
    assert_almost_equal(baseline, got)

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


def benchmark(sizes, config, itype, times_col: int = 1, times_indices: int = 1):

    data = []
    for size in tqdm(sizes):

        if config == "llama":
            # zeros: 32000x4096
            # indices: 2x1024x1
            # updates: 2x1024x4096
            shape = (32000, 4096)
            shape_indices = (2, size, 1)
        else:
            shape = (size, int(size * times_col))
            shape_indices = (2, int(size * times_indices), 1)
        shape_updates = (2, shape_indices[1], shape[-1])

        shape = np.array(shape, dtype=np.int64)
        indices = np.array(
            [np.random.randint(-1, shape[0]) for _ in range(np.prod(shape_indices))],
            dtype=np.int64,
        ).reshape(shape_indices)
        updates = np.random.randn(*shape_updates).astype(
            np.float32 if itype == TensorProto.FLOAT else np.float16
        )
        feeds = dict(shape=shape, indices=indices, updates=updates)

        for op_type in ["ScatterND", "ScatterNDOfShape", "MaskedScatterNDOfShape"]:
            model = get_model(op_type)
            sess = get_session(model)
            bind, _cuda_feeds = move_inputs(sess, feeds)
            begin = time.perf_counter()
            for _i in range(script_args.warmup):
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
                label=op_type,
                warmup=warmup,
                time=npt.mean(),
                std=npt.std(),
                min=npt.min(),
                max=npt.max(),
                repeat=script_args.repeat,
                size=size,
            )
            data.append(obs)
    return data


#######################################
# Benchmark.


if sess is not None:

    print(f"sizes={sizes}")

    data_nd = benchmark(sizes, script_args.config, itype=itype, times_col=2)

##########################################
# Data
# ++++

if sess is not None:

    df = DataFrame(data_nd)
    df.to_csv("plot_op_scatternd_mask_cuda.csv", index=False)
    df.to_csv("plot_op_scatternd_mask_cuda.xlsx", index=False)
    print(df.head())

#####################
# Pivot.

if sess is not None:

    pivot = df.pivot(index="size", columns="label", values="time")
    col = pivot["ScatterND"].copy()
    print("Time")
    print(pivot)
    for c in pivot.columns:
        pivot[c] = col / pivot[c]
    print("Speed up compare to the onnx standaed.")
    print(pivot)

    ax = pivot.plot(
        logx=True,
        logy=True,
        title=f"Optimization for ScatterND on CUDA\nitype={itype}",
    )
    ax.get_figure().savefig("plot_op_scatternd_mask_cuda.png")

##############################
# It requires more test to determine when it is better.
# But the fused operator with mask seems more efficient in any case
# compare to the fused operator without mask.
# For big sizes, ScatterND seems very slow as it is using atomic addition.
