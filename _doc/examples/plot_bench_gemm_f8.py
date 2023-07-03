"""
.. _l-example-bench-gemm-f8:

Measuring Gemm performance with different input and output types
================================================================

This benchmark looks into various combinations allowed by functions
:epkg:`cublasLtMatmul`.
"""
import pprint
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx_extended.ext_test_case import unit_test_going

try:
    from onnx_extended.validation.cuda.cuda_example_py import (
        gemm_benchmark_test,
        get_device_prop,
    )

    has_cuda = True
except ImportError:
    # CUDA not available.
    has_cuda = False

#############################################
# Device
# ++++++

if has_cuda:
    prop = get_device_prop()
    pprint.pprint(prop)
else:
    print("CUDA is not available")
    prop = dict(major=0)


#############################################
# Configurations
# ++++++++++++++

if prop["major"] <= 0:
    # No CUDA.
    tests = []
    dims = []
elif prop["major"] < 7:
    # No float 8.
    tests = list(range(5))
    dims = [16, 32, 64]
elif prop["major"] < 9:  # T100, A100
    # No float 8.
    tests = list(range(5))
    dims = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
else:
    tests = list(range(15))  # H100
    dims = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

##############################
# Benchmark
# +++++++++


def type2string(dt):
    dtypes = {0: "F32", 2: "F16", 14: "BF16", 28: "E4M3", 29: "E5M2"}
    return dtypes[int(dt)]


pbar = tqdm(list(product(tests, dims)))
obs = []
for test, dim in pbar:
    pbar.set_description(f"test={test} dim={dim}")
    if test in {8, 9, 10, 12, 13}:
        # not valid yet
        continue
    if dim < 128:
        n, N = 20, 100
    elif dim < 512:
        n, N = 20, 50
    elif dim < 8192:
        n, N = 10, 25
    else:
        n, N = 3, 5

    # warmup
    gemm_benchmark_test(test, n, dim)

    # benchmark
    res = gemm_benchmark_test(test, N, dim)

    # better rendering
    res["test"] = test
    update = {}
    for k, v in res.items():
        if "type_" in k:
            update[k] = type2string(v)
        if k.startswith("t-"):
            update[k] = res[k] / res["N"]
    update["compute_type"] = f"C{int(res['compute_type'])}"
    update["N"] = int(res["N"])
    update["dim"] = int(res["dim"])
    update["name"] = (
        f"{update['type_a']}x{update['type_b']}->"
        f"{update['type_d']}{update['compute_type']}"
    )
    res.update(update)
    obs.append(res)
    if unit_test_going() and len(obs) > 2:
        break

df = DataFrame(obs)
df.to_csv("plot_bench_gemm_f8.csv", index=False)
df.to_excel("plot_bench_gemm_f8.xlsx", index=False)
print(df.head().T)

df.head().T

###################################
# Test definition
# +++++++++++++++

col_def = ["name", "test", "type_a", "type_b", "type_d", "compute_type"]
if df.shape[0] > 0:
    deft = df.copy()
    gr = deft[col_def].groupby(col_def, as_index=False).count()
    print(gr)

###################################
# Total time and only gemm
# ++++++++++++++++++++++++

if df.shape[0] > 0:
    dfi = df[col_def + ["dim", "t-total", "t-gemm_sync"]]
    print(dfi)

###################################
# Smaller sets
# ++++++++++++

if df.shape[0] > 0:
    subset = {1, 3, 4, 5, 7}
    dfis = dfi[dfi.test.isin(subset)]
    print()
    print("t-gemm_sync")
    pivi = dfis.pivot_table(index="dim", columns="name", values="t-gemm_sync")
    print(pivi)
    print()
    print("t-total")
    pivi = dfis.pivot_table(index="dim", columns="name", values="t-total")
    print(pivi)


###################################
# Plots
# +++++

if df.shape[0] > 0:
    piv = df.pivot_table(index="dim", columns="name", values="t-gemm_sync")
    piv.plot(title="MatMul performances")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    piv.plot(ax=ax[0], title="Gemm performance\nlower is better", logx=True, logy=True)

    piv = df[df.test.isin(subset)].pivot_table(
        index="dim", columns="name", values="t-gemm_sync"
    )
    piv.plot(ax=ax[1], title="Gemm performance\nlower is better", logx=True, logy=True)

    fig.tight_layout()
    fig.savefig("plot_bench_gemm_f8.png")
