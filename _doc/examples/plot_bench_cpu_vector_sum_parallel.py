"""
.. _l-example-bench-cpu-vector-sum:

Measuring CPU performance with a parallelized vector sum
========================================================

The example compares the time spend in computing the sum of all
coefficients of a matrix when the function walks through the coefficients
by rows or by columns when the computation is parallelized.

Vector Sum
++++++++++
"""
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx_extended.ext_test_case import measure_time, unit_test_going
from onnx_extended.validation.cpu._validation import (
    vector_sum_array as vector_sum,
    vector_sum_array_parallel as vector_sum_parallel,
)

obs = []
dims = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000]
if unit_test_going():
    dims = dims[:3]
for dim in tqdm(dims):
    values = numpy.ones((dim, dim), dtype=numpy.float32).ravel()
    diff = abs(vector_sum(dim, values, True) - dim**2)

    res = measure_time(lambda: vector_sum(dim, values, True), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="rows",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    res = measure_time(lambda: vector_sum_parallel(dim, values, True), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="rows//",
            time_per_element=res["average"] / dim**2,
            diff=diff,
        )
    )

    diff = abs(vector_sum(dim, values, False) - dim**2)
    res = measure_time(lambda: vector_sum_parallel(dim, values, False), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            direction="cols//",
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
fig.savefig("plot_bench_cpu_vector_sum_parallel.png")

##############################################
# The summation by rows is much faster as expected.
# That explains why it is usually more efficient to
# transpose the first matrix before a matrix multiplication.
# Parallelization is faster.
