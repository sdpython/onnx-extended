"""
.. _l-example-plot-onnx2-time:

Measures loading, saving time for an onnx model in python
=========================================================

"""

import os
import time
import numpy as np
import onnx
import onnx_extended.onnx2 as onnx2


onnx_file = (
    "dump_test/microsoft_Phi-4-mini-reasoning-onnx-dynamo-ir/"
    "microsoft_Phi-4-mini-reasoning-onnx-dynamo-ir.onnx"
)
if not os.path.exists(onnx_file):
    print("Creates the model, starts with importing transformers...")
    import torch  # noqa: F401
    import transformers  # noqa: F401

    print("Imports onnx-diagnostic...")
    from onnx_diagnostic.torch_models.validate import validate_model

    print("Starts creating the model...")

    validate_model(
        "microsoft/Phi-4-mini-reasoning",
        do_run=True,
        verbose=2,
        exporter="onnx-dynamo",
        do_same=True,
        patch=True,
        rewrite=True,
        optimization="ir",
        dump_folder="dump_test",
    )

    print("done.")

# %%
# Let's load and save the model to get one unique file.

full_name = "dump_test/microsoft_Phi-4-mini-reasoning.onnx"
if not os.path.exists(full_name):
    print("Loads the model and saves it as one unique file.")
    onx = onnx.load(onnx_file)
    onnx.save(onx, full_name)

# %%
# Let's get the size.


size = os.stat(full_name).st_size
print(f"model size {size / 2**20:1.3f} Mb")

# %%
# Measures the loading time
# +++++++++++++++++++++++++


def measure(f, N=3):
    times = []
    for _ in range(N):
        begin = time.perf_counter()
        onx = f()
        end = time.perf_counter()
        times.append(end - begin)
    return onx, {"avg": np.mean(times), "times": times}


# %%
# Let's do it with onnx2.

print("Loading time with onnx2.")
onx2, times = measure(lambda: onnx2.load(full_name))
print(times)

# %%
# Then with onnx.

print("Loading time with onnx.")
onx, times = measure(lambda: onnx.load(full_name))
print(times)

# %%
# Let's do it with onnx2 but the loading of the tensors is parallelized.

print(
    f"Loading time with onnx2 and 4 threads, "
    f"it has {len(onx2.graph.initializer)} initializers"
)
onx2, times = measure(lambda: onnx2.load(full_name, parallel=True, num_threads=4))
print(times)

# %%
# It looks much faster.

# %%
# Let's load it with :epkg:`onnxruntime`.
import onnxruntime

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
print("Loading time with onnxruntime")
_, times = measure(
    lambda: onnxruntime.InferenceSession(
        full_name, so, providers=["CPUExecutionProvider"]
    )
)
print(times)


# %%
# Measure the saving time
# +++++++++++++++++++++++
#
# Let's do it with onnx2.

print("Saving time with onnx2.")
_, times = measure(lambda: onnx2.save(onx2, full_name))
print(times)

# %%
# Then with onnx.

print("Saving time with onnx.")
_, times = measure(lambda: onnx.save(onx, full_name))
print(times)

# %%
# Measure the saving time with external weights
# +++++++++++++++++++++++++++++++++++++++++++++
#
# Let's do it with onnx2.

full_name = "dump_test/microsoft_Phi-4-mini-reasoning.ext.onnx"
full_weight = "dump_test/microsoft_Phi-4-mini-reasoning.ext.data"

print("Saving time with onnx2 and external weights.")
_, times = measure(lambda: onnx2.save(onx2, full_name, location=full_weight))
print(times)

# %%
# Then with onnx. We can only do that once,
# the function modifies the model inplace to add information
# about external data. The second run does not follow the same steps.

print("Saving time with onnx and external weights.")
full_weight += ".2"
_, times = measure(
    lambda: onnx.save(
        onx,
        full_name,
        location=os.path.split(full_weight)[-1],
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    ),
    N=1,
)

print(times)
