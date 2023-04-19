"""
.. _l-example-bench-gpu-vector:

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
from onnx_extended.validation._validation import (
    vector_sum_array_avx as vector_sum_avx,
    vector_sum_array_avx_parallel as vector_sum_avx_parallel,
)

try:
    from onnx_extended.validation.cuda_example_py import (
        vector_sum0,
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

    diff = abs(vector_sum0(values, 128) - dim**2)
    res = measure_time(lambda: vector_sum0(values, 128), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="0cuda128",
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
fig.savefig("plot_bench_cpu_vector_sum_avx_parallel.png")

##############################################
# AVX is faster.
