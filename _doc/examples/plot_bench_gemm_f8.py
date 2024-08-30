"""
.. _l-example-gemm-f8:

Measuring Gemm performance with different input and output tests
================================================================

This benchmark looks into various combinations allowed by functions
:epkg:`cublasLtMatmul`. The tested configurations are available at
:epkg:`cuda_gemm.cu`.
"""

import pprint
import warnings
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from onnx_extended.args import get_parsed_args
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
    gemm_benchmark_test = None

if has_cuda:
    prop = get_device_prop()
    if prop["major"] <= 0:
        # No CUDA.
        dtests, ddims = "", ""
    elif prop["major"] < 7:
        # No float 8.
        dtests, ddims = "0,1,2,3,4,15", "16,32,64,64x128x92"
    elif prop["major"] < 9:  # T100, A100
        # No float 8.
        dtests, ddims = (
            "0,1,2,3,4,15",
            "16,32,64,128,256,512,1024,2048,4096,8192,"
            "128x768x768,128x3072x768,128x768x3072",
        )
    else:
        dtests, ddims = (
            "0,1,2,3,4,5,6,7,11,14,15",
            "16,32,64,128,256,512,1024,2048,4096,8192,16384,"
            "128x768x768,128x3072x768,128x768x3072",
        )
else:
    dtests, ddims = "", ""


script_args = get_parsed_args(
    "plot_bench_gemm_f8",
    description=__doc__,
    dims=(
        "16,32" if unit_test_going() else ddims,
        "square matrix dimensions to try, comma separated values",
    ),
    tests=(
        "0,1,2" if unit_test_going() else dtests,
        "configuration to check, see cuda_gemm.cu",
    ),
    warmup=2 if unit_test_going() else 5,
    repeat=2 if unit_test_going() else 10,
    expose="repeat,warmup",
)

#############################################
# Device
# ++++++

if has_cuda:
    prop = get_device_prop()
    pprint.pprint(prop)
else:
    print("CUDA is not available")
    prop = dict(major=0)


##############################
# Benchmark
# +++++++++


def type2string(dt):
    dtests = {
        0: "F32",
        2: "F16",
        14: "BF16",
        28: "E4M3",
        29: "E5M2",
        3: "I8",
        10: "I32",
    }
    return dtests[int(dt)]


dims = []
tests = []
if gemm_benchmark_test is not None:
    for d in script_args.dims.split(","):
        if "x" in d:
            spl = d.split("x")
            m, n, k = tuple(int(i) for i in spl)
            dims.append((m, n, k))
        else:
            dims.append(int(d))
    tests = [int(i) for i in script_args.tests.split(",")]

pbar = tqdm(list(product(tests, dims)))
obs = []
for test, dim in pbar:
    pbar.set_description(f"type={test} dim={dim}")
    if test in {8, 9, 10, 12, 13}:
        warnings.warn(f"unsupported configuration {test}.", stacklevel=0)
        continue
    mdim = dim if isinstance(dim, int) else max(dim)
    if mdim < 128:
        n, N = script_args.warmup * 8, script_args.repeat * 8
    elif mdim < 512:
        n, N = script_args.warmup * 4, script_args.repeat * 4
    elif mdim < 8192:
        n, N = script_args.warmup * 2, script_args.repeat * 2
    else:
        n, N = script_args.warmup, script_args.repeat

    if isinstance(dim, int):
        gemm_args = [dim] * 6
    else:
        m, n, k = dim
        lda, ldb, ldd = k, k, k
        gemm_args = [m, n, k, lda, ldb, ldd]

    # warmup
    gemm_benchmark_test(test, N, *gemm_args)

    # benchmark
    res = gemm_benchmark_test(test, N, *gemm_args)

    # better rendering
    res["test"] = test
    update = {}
    for k, v in res.items():
        if "type_" in k:
            update[k] = type2string(v)
        if k.startswith("t-"):
            update[k] = res[k] / res["N"]
    update["compute_type"] = f"C{int(res['compute_type'])}"
    for c in ["N", "m", "n", "k", "lda", "ldb", "ldd"]:
        update[c] = int(res[c])
    update["~dim"] = (update["k"] * max(update["m"], update["n"])) ** 0.5
    update["mnk"] = f"{update['m']}x{update['n']}x{update['k']}"
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
    dfi = df[[*col_def, "~dim", "mnk", "t-total", "t-gemm_sync"]]
    print(dfi)

###################################
# Smaller sets
# ++++++++++++

if df.shape[0] > 0:
    subset = {1, 3, 4, 5, 7}
    dfis = dfi[dfi.test.isin(subset)]
    print()
    print("t-gemm_sync")
    pivi = dfis.pivot_table(index=["~dim", "mnk"], columns="name", values="t-gemm_sync")
    print(pivi)
    print()
    print("t-total")
    pivi = dfis.pivot_table(index=["~dim", "mnk"], columns="name", values="t-total")
    print(pivi)


###################################
# Plots
# +++++

if df.shape[0] > 0:
    piv = df.pivot_table(index=["~dim", "mnk"], columns="name", values="t-gemm_sync")
    piv.plot(title="MatMul performances")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    piv.plot(ax=ax[0], title="Gemm performance\nlower is better", logx=True, logy=True)

    piv = df[df.test.isin(subset)].pivot_table(
        index=["~dim", "mnk"], columns="name", values="t-gemm_sync"
    )
    if piv.shape[0] > 0:
        piv.plot(
            ax=ax[1], title="Gemm performance\nlower is better", logx=True, logy=True
        )

    fig.tight_layout()
    fig.savefig("plot_bench_gemm_f8.png")
