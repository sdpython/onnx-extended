"""
Measuring CPU/GPU performance with a vector sum
===============================================

The examples compares multiple versions of a vector sum,
CPU, GPU.

Vector Sum
++++++++++
"""
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx_extended.ext_test_case import measure_time, unit_test_going
from onnx_extended.validation.cpu._validation import (
    vector_sum_array_avx as vector_sum_avx,
    vector_sum_array_avx_parallel as vector_sum_avx_parallel,
)

try:
    from onnx_extended.validation.cuda.cuda_example_py import (
        vector_sum0,
        vector_sum6,
        vector_sum_atomic,
    )
except ImportError:
    # CUDA is not available
    vector_sum0 = None

obs = []
dims = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000]
if unit_test_going():
    dims = dims[:3]
for dim in tqdm(dims):
    values = numpy.ones((dim, dim), dtype=numpy.float32).ravel()

    diff = abs(vector_sum_avx(dim, values) - dim**2)
    res = measure_time(lambda: vector_sum_avx(dim, values), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="avx",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    diff = abs(vector_sum_avx_parallel(dim, values) - dim**2)
    res = measure_time(lambda: vector_sum_avx_parallel(dim, values), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="avx//",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    if vector_sum0 is None:
        # CUDA is not available
        continue

    diff = abs(vector_sum0(values, 32) - dim**2)
    res = measure_time(lambda: vector_sum0(values, 32), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="0cuda32",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    diff = abs(vector_sum_atomic(values, 32) - dim**2)
    res = measure_time(lambda: vector_sum_atomic(values, 32), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="Acuda32",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    diff = abs(vector_sum6(values, 32) - dim**2)
    res = measure_time(lambda: vector_sum6(values, 32), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="6cuda32",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    diff = abs(vector_sum6(values, 256) - dim**2)
    res = measure_time(lambda: vector_sum6(values, 256), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="6cuda256",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

df = DataFrame(obs)
piv = df.pivot(index="dim", columns="direction", values="time_per_element")
print(piv)


##############################################
# Plots
# +++++

piv_diff = df.pivot(index="dim", columns="direction", values="diff")
piv_time = df.pivot(index="dim", columns="direction", values="time")

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
piv.plot(ax=ax[0], logx=True, title="Comparison between two summation")
piv_diff.plot(ax=ax[1], logx=True, logy=True, title="Summation errors")
piv_time.plot(ax=ax[2], logx=True, logy=True, title="Total time")
fig.savefig("plot_bench_gpu_vector_sum_gpu.png")

##############################################
# The results should look like the following.
#
# .. image:: ../_static/vector_sum6_results.png
#
# AVX is still faster. Let's try to understand why.
#
# Profiling
# +++++++++
#
# The profiling indicates where the program is most of the time.
# It shows when the GPU is waiting and when the memory is copied from
# from host (CPU) to device (GPU) and the other way around. There are
# the two steps we need to reduce or avoid to make use of the GPU.
#
# Profiling with `nsight-compute <https://developer.nvidia.com/nsight-compute>`_:
#
# ::
#
#     nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx,openmp python <file>
#
# If `nsys` fails to find `python`, the command `which python` should locate it.
# `<file> can be `plot_bench_gpu_vector_sum_gpu.py` for example.
#
# Then command `nsys-ui` starts the Visual Interface interface of the profiling.
# A screen shot shows the following after loading the profiling.
#
# .. image:: ../_static/vector_sum6.png
#
# Most of time is spent in copy the data from CPU memory to GPU memory.
# In our case, GPU is not really useful because just copying the data from CPU
# to GPU takes more time than processing it with CPU and AVX instructions.
#
# GPU is useful for deep learning because many operations can be chained and
# the data stays on GPU memory until the very end. When multiple tools are involved,
# torch, numpy, onnxruntime, the `DLPack <https://github.com/dmlc/dlpack>`_
# avoids copying the data when switching.
#
# The copy of a big tensor can happens by block. The computation may start
# before the data is fully copied.
