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
    from onnx_diagnostic.torch_models.validate import validate_model

    print("Creates the model...")

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

print("Load time with onnx2.")
onx2, times = measure(lambda: onnx2.load(full_name))
print(times)

# %%
# Then with onnx.

print("Load time with onnx.")
onx, times = measure(lambda: onnx.load(full_name))
print(times)

# %%
# Measure the saving time
# +++++++++++++++++++++++
#
# Let's do it with onnx2.

print("Save time with onnx2.")
_, times = measure(lambda: onnx2.save(onx2, full_name))
print(times)

# %%
# Then with onnx.

print("Save time with onnx.")
_, times = measure(lambda: onnx.save(onx, full_name))
print(times)
