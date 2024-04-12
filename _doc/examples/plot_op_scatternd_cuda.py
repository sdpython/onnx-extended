"""
.. _l-example-op-scatternd_cuda:

=====================================
Optimizing ScatterND operator on CUDA
=====================================

How to parallelize something like the following?

ScatterND
=========

This configuration happens in a :epkg:`LLAMA` model.

::

    gradient = ScatterND(zeros, indices, updates)

Where the shapes are:

* zeros: 32000x4906
* indices: 2x1024x1
* updates: 2x1024x4096
"""

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


def get_model(d3=True):
    indices_shape = ["i", "j", 1] if d3 else ["m", 1]
    updates_shape = ["i", "j", "b"] if d3 else ["m", "b"]
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "ScatterND", ["X", "indices", "updates"], ["Y"], reduction="add"
                )
            ],
            "g",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b"]),
                oh.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                oh.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", "b"])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    return model


model = get_model()
print(onnx_simple_text_plot(model))


##########################################
# Let's see the evaluation by the ReferenceEvaluator.


def _scatter_nd_impl(data, indices, updates, reduction=None):  # type: ignore
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
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
    def _run(self, data, indices, updates, reduction=None):  # type: ignore
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


shape = (5, 7)
X = np.zeros(shape, dtype=np.float32)
indices = np.zeros((2, 10, 1)).astype(np.int64)
indices[:, ::2, 0] = 3
updates = np.ones((2, 10, 7)).astype(np.float32)
feeds = {"X": X, "indices": indices, "updates": updates}


ref = ReferenceEvaluator(model, new_ops=[ScatterND])
got = ref.run(None, feeds)[0]
print(got)


###########################################
# To generalize, let's change the shapes.

model = get_model(d3=False)
print(onnx_simple_text_plot(model))


new_indices = indices.reshape((-1, 1))
new_updates = updates.reshape((-1, updates.shape[-1]))
feeds = {"X": X, "indices": indices, "updates": updates}

ref = ReferenceEvaluator(model, new_ops=[ScatterND])
got = ref.run(None, feeds)[0]
print(got)
