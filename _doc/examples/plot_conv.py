"""
.. _l-example-conv:

Using C implementation of operator Conv
=======================================

*onnx-extended* includes an implementation of operator Conv
in language C++ must faster than the python implementation
available in package :epkg:`onnx`. These implementations
are automatically available through class
:class:`onnx_extended.reference.CReferenceEvaluator`.
The following example compares the processing time for three runtimes.

Creation of a simple model
++++++++++++++++++++++++++
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession
from onnx_extended.ext_test_case import measure_time, unit_test_going
from onnx_extended.reference import CReferenceEvaluator


X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None, None, None])
W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
node = make_node(
    "Conv",
    ["X", "W", "B"],
    ["Y"],
    pads=[1, 1, 1, 1],
    dilations=[1, 1],
    strides=[2, 2],
)
graph = make_graph([node], "g", [X, W, B], [Y])
onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)], ir_version=8)

#########################################
# ReferenceEvaluator and CReferenceEvaluator
# ++++++++++++++++++++++++++++++++++++++++++
# Let's first compare the outputs are the same.

sH, sW = 64, 64
X = np.arange(sW * sH).reshape((1, 1, sH, sW)).astype(np.float32)
W = np.ones((1, 1, 3, 3), dtype=np.float32)
B = np.array([[[[0]]]], dtype=np.float32)

sess1 = ReferenceEvaluator(onnx_model)
sess2 = CReferenceEvaluator(onnx_model)

expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
diff = np.abs(expected - got).max()
print(f"difference: {diff}")

#####################################################
# Everything works fine.
#
# Time measurement
# ++++++++++++++++

feeds = {"X": X, "W": W, "B": B}

t1 = measure_time(lambda: sess1.run(None, feeds))
print(f"ReferenceEvaluator: {t1['average']}s")

t2 = measure_time(lambda: sess2.run(None, feeds))
print(f"CReferenceEvaluator: {t2['average']}s")
print(f"speedup is {t1['average'] / t2['average']}")

############################################
# Let's add :epkg:`onnxruntime` as well.

sess3 = InferenceSession(
    onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
)

t3 = measure_time(lambda: sess3.run(None, feeds))
print(f"InferenceSession: {t3['average']}s")
print(f"speedup is {t1['average'] / t3['average']}")


############################################
# Plotting
# ++++++++

data = []

for i in tqdm([16, 32, 48, 64]):
    sH, sW = i, i
    X = np.arange(sW * sH).reshape((1, 1, sH, sW)).astype(np.float32)
    W = np.ones((1, 1, 3, 3), dtype=np.float32)
    B = np.array([[[[0]]]], dtype=np.float32)
    feeds = {"X": X, "W": W, "B": B}
    t1 = measure_time(lambda: sess1.run(None, feeds))
    t2 = measure_time(lambda: sess2.run(None, feeds))
    obs = dict(size=i, onnx=t1["average"], onnx_extended=t2["average"])
    data.append(obs)
    if unit_test_going() and len(data) >= 2:
        break

df = DataFrame(data)
df


##########################################
# Finally.

df = df.set_index("size")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
df.plot(
    ax=ax, logx=True, logy=True, title="Comparison python / C implementation for Conv"
)
df["speedup"] = df["onnx"] / df["onnx_extended"]
ax2 = ax.twinx()
df[["speedup"]].plot(ax=ax2, color="green")

fig.savefig("plot_conv.png")
# plt.show()
