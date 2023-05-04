"""
.. _l-example-conv-denorm:

How float format has an impact on speed computation
===================================================

An example with Conv.

Imports a specific model
++++++++++++++++++++++++
"""
import os
import urllib.request as ur
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from onnx import load
from onnx.helper import make_graph, make_model
from onnx.numpy_helper import to_array, from_array
from onnxruntime import (
    InferenceSession,
    get_all_providers,
    OrtValue,
    SessionOptions,
    GraphOptimizationLevel,
)
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_extended.ext_test_case import measure_time, unit_test_going
from onnx_extended.reference import CReferenceEvaluator

try:
    import torch
except ImportError:
    # no torch is available
    print("torch is not available")
    torch = None

name = "slow_conv.zip"
if not os.path.exists(name):
    print("download {url!r}")
    url = f"https://github.com/microsoft/onnxruntime/files/11388184/{name}"
    ur.urlretrieve(url, name)

onnx_file = "slow_conv.onnx"
if not os.path.exists(onnx_file):
    print("unzip f{onnx_file}")
    with ZipFile(name) as z:
        print(z.namelist())
        with z.open(z.namelist()[0]) as f:
            with open(name, "wb") as g:
                g.write(f.read())

###################################################
# The model looks like:

with open(onnx_file, "rb") as f:
    onx = load(f)

print(onnx_simple_text_plot(onx))


onnx_model = onnx_file
input_shape = (1, 256, 14, 14)
X = np.ones(input_shape, dtype=np.float32)

#########################################
# CReferenceEvaluator and InferenceSession
# ++++++++++++++++++++++++++++++++++++++++
# Let's first compare the outputs are the same.

sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL


sess1 = CReferenceEvaluator(onnx_model)
sess2 = InferenceSession(onnx_model, sess_options, providers=["CPUExecutionProvider"])

expected = sess1.run(None, {"X": X})[0]
got = sess2.run(None, {"X": X})[0]
diff = np.abs(expected - got).max()
print(f"difference: {diff}")

#####################################################
# Everything works fine.
#
# Time measurement
# ++++++++++++++++

feeds = {"X": X}

t1 = measure_time(lambda: sess1.run(None, feeds), repeat=2, number=5)
print(f"CReferenceEvaluator: {t1['average']}s")

t2 = measure_time(lambda: sess2.run(None, feeds), repeat=2, number=5)
print(f"InferenceSession: {t2['average']}s")

############################################
# Plotting
# ++++++++
#
# Let's modify the the weight of the model and multiply everything by a scalar.
# Let's choose an random input.
has_cuda = "CUDAExecutionProvider" in get_all_providers()
X = np.random.random(X.shape).astype(X.dtype)


def modify(onx, scale):
    t = to_array(onx.graph.initializer[0])
    b = to_array(onx.graph.initializer[1]).copy()
    t = (t * scale).astype(np.float32)
    graph = make_graph(
        onx.graph.node,
        onx.graph.name,
        onx.graph.input,
        onx.graph.output,
        [from_array(t, name=onx.graph.initializer[0].name), onx.graph.initializer[1]],
    )
    return t, b, make_model(graph, opset_imports=onx.opset_import)


scales = [2**p for p in range(0, 31, 2)]
data = []
feeds = {"X": X}
expected = sess2.run(None, feeds)[0]
if torch is not None:
    tx = torch.from_numpy(X)

sess_options0 = SessionOptions()
sess_options0.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
sess_options0.add_session_config_entry("session.set_denormal_as_zero", "1")

for scale in tqdm(scales):
    w, b, new_onx = modify(onx, scale)
    # sess1 = CReferenceEvaluator(new_onx)
    sess2 = InferenceSession(
        new_onx.SerializeToString(), sess_options, providers=["CPUExecutionProvider"]
    )
    sess3 = InferenceSession(
        new_onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess4 = InferenceSession(
        new_onx.SerializeToString(), sess_options0, providers=["CPUExecutionProvider"]
    )

    # sess1.run(None, feeds)
    got = sess2.run(None, feeds)[0]
    diff = np.abs(got / scale - expected).max()
    sess3.run(None, feeds)
    got0 = sess4.run(None, feeds)[0]
    diff0 = np.abs(got0 / scale - expected).max()

    # t1 = measure_time(lambda: sess1.run(None, feeds), repeat=2, number=5)
    t2 = measure_time(lambda: sess2.run(None, feeds), repeat=2, number=5)
    t3 = measure_time(lambda: sess3.run(None, feeds), repeat=2, number=5)
    t4 = measure_time(lambda: sess4.run(None, feeds), repeat=2, number=5)
    obs = dict(
        scale=scale, ort=t2["average"], diff=diff, diff0=diff0, ort0=t4["average"]
    )
    # obs["ref"]=t1["average"]
    obs["ort-opt"] = t3["average"]

    if torch is not None:
        tw = torch.from_numpy(w)
        tb = torch.from_numpy(b)
        torch.nn.functional.conv2d(tx, tw, tb, padding=1)
        t3 = measure_time(
            lambda: torch.nn.functional.conv2d(tx, tw, tb, padding=1),
            repeat=2,
            number=5,
        )
        obs["torch"] = t3["average"]

    if has_cuda:
        sess2 = InferenceSession(
            new_onx.SerializeToString(),
            sess_options,
            providers=["CUDAExecutionProvider"],
        )
        sess3 = InferenceSession(
            new_onx.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        x_ortvalue = OrtValue.ortvalue_from_numpy(X, "cuda", 0)
        cuda_feeds = {"X": x_ortvalue}
        sess2.run_with_ort_values(None, cuda_feeds)
        sess3.run_with_ort_values(None, cuda_feeds)
        t2 = measure_time(lambda: sess2.run(None, cuda_feeds), repeat=2, number=5)
        t3 = measure_time(lambda: sess3.run(None, cuda_feeds), repeat=2, number=5)
        obs["ort-cuda"] = t2["average"]
        obs["ort-cuda-opt"] = t2["average"]

    data.append(obs)
    if unit_test_going() and len(data) >= 3:
        break

df = DataFrame(data)
df

print(df)

##########################################
# Finally.

df = df.drop(["diff", "diff0"], axis=1).set_index("scale")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
df.plot(ax=ax, logx=True, logy=True, title="Comparison Conv in a weird case")

fig.savefig("plot_conv_denorm.png")
# plt.show()


##########################################
# Conclusion
# ++++++++++
#
#
