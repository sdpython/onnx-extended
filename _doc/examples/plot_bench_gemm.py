"""
.. _l-example-bench-gemm:

Measuring performance about Gemm
================================

Differents types, differents backend, differents

Onnx Model
++++++++++
"""
import platform
from itertools import product
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame, pivot_table
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnx.numpy_helper import from_array
from onnxruntime import InferenceSession, get_available_providers
from onnxruntime.capi._pybind_state import (
    OrtValue as C_OrtValue,
    OrtDevice as C_OrtDevice,
)
from onnxruntime.capi.onnxruntime_pybind11_state import (
    NotImplemented,
    InvalidGraph,
    InvalidArgument,
)
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ext_test_case import unit_test_going, measure_time


def create_model(mat_type=TensorProto.FLOAT):
    I1 = from_array(numpy.array([1], dtype=numpy.float32), name="I")
    A = make_tensor_value_info("A", mat_type, [None, None])
    B = make_tensor_value_info("B", mat_type, [None, None])
    C = make_tensor_value_info("C", mat_type, [None, None])
    nodes = [
        make_node("CastLike", ["I", "A"], ["Ic"]),
        make_node("Add", ["A", "Ic"], ["A1"]),
        make_node("Add", ["A1", "Ic"], ["A2"]),
        make_node("Add", ["A2", "Ic"], ["A3"]),
        make_node("MatMul", ["A", "B"], ["M0"]),
        make_node("MatMul", ["A1", "B"], ["M1"]),
        make_node("MatMul", ["A2", "B"], ["M2"]),
        make_node("MatMul", ["A3", "B"], ["M3"]),
        make_node("Add", ["M0", "M1"], ["M12"]),
        make_node("Add", ["M2", "M3"], ["M23"]),
        make_node("Add", ["M12", "M23"], ["C"]),
    ]
    graph = make_graph(nodes, "a", [A, B], [C], [I1])
    if mat_type < 16:
        # regular type
        opset, ir = 18, 8
    else:
        opset, ir = 19, 9
    onnx_model = make_model(
        graph, opset_imports=[make_opsetid("", opset)], ir_version=ir
    )
    check_model(onnx_model)
    return onnx_model


create_model()

###########################################
# A model to cast


def create_cast(to):
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    C = make_tensor_value_info("C", to, [None, None])
    node1 = make_node("Cast", ["A"], ["C"], to=to)
    graph = make_graph([node1], "a", [A], [C])
    if to < 16:
        # regular type
        opset, ir = 18, 8
    else:
        opset, ir = 19, 9
    onnx_model = make_model(
        graph, opset_imports=[make_opsetid("", opset)], ir_version=ir
    )
    check_model(onnx_model)
    return onnx_model


create_cast(TensorProto.FLOAT16)


##############################
# Performance
# +++++++++++
#
# The benchmark will run the following configurations.

types = [
    TensorProto.FLOAT,
    TensorProto.UINT32,
    TensorProto.INT32,
    TensorProto.INT16,
    TensorProto.INT8,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT8E5M2,
]
engine = [CReferenceEvaluator, InferenceSession]
providers = [
    ["CPUExecutionProvider"],
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
]
# M, N, K
dims = [
    (10, 10, 10),
    (61, 62, 63),
    (64, 64, 64),
    (65, 66, 67),
    (100, 100, 100),
    (128, 128, 128),
    # (256, 256, 256),
    # (400, 400, 400),
    # (512, 512, 512),
]


map_type = {TensorProto.FLOAT: numpy.float32, TensorProto.FLOAT16: numpy.float16}


####################################
# Let's cache the matrices involved.


def to_ort_value(m):
    device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    ort_value = C_OrtValue.ortvalue_from_numpy(m, device)
    return ort_value


matrices = {}
for m, n, k in dims:
    for tt in types:
        for i, j in [(m, k), (k, n)]:
            try:
                sess = InferenceSession(
                    create_cast(tt).SerializeToString(),
                    providers=["CPUExecutionProvider"],
                )
            except (InvalidGraph, InvalidArgument, NotImplemented):
                # not support by this version of onnxruntime
                continue
            vect = (numpy.random.randn(i, j) * 10).astype(numpy.float32)
            ov = to_ort_value(vect)
            ovtt = sess._sess.run_with_ort_values({"A": ov}, ["C"], None)[0]
            matrices[tt, i, j] = ovtt

print(f"{len(matrices)} matrices were created.")

###################################
# Let's run the benchmark


data = []
errors = []
pbar = tqdm(list(product(types, engine, providers, dims)))
for tt, engine, provider, dim in pbar:
    if max(dim) <= 200:
        repeat, number = 50, 25
    elif max(dim) <= 256:
        repeat, number = 25, 10
    else:
        repeat, number = 10, 4

    onx = create_model(tt)
    with open(f"plot_bench_gemm_{tt}.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    k1 = (tt, dim[0], dim[2])
    k2 = (tt, dim[2], dim[1])
    if k1 not in matrices:
        errors.append(f"Key k1={k1!r} not in matrices.")
        continue
    if k2 not in matrices:
        errors.append(f"Key k2={k2!r} not in matrices.")
        continue

    if engine == CReferenceEvaluator:
        if tt == TensorProto.FLOAT16 and max(dim) > 50:
            repeat, number = 2, 2
        if provider != ["CPUExecutionProvider"]:
            continue
        if tt not in [TensorProto.FLOAT, TensorProto.FLOAT16]:
            continue

        pbar.set_description(
            f"t={tt} e={engine.__name__} p={provider[0][:4]} dim={dim}"
        )

        feeds = {"A": matrices[k1].numpy(), "B": matrices[k2].numpy()}
        sess = engine(onx)
        sess.run(None, feeds)
        obs = measure_time(lambda: sess.run(None, feeds), repeat=repeat, number=number)

    elif engine == InferenceSession:
        if provider[0] not in get_available_providers():
            continue
        pbar.set_description(
            f"t={tt} e={engine.__name__} p={provider[0][:4]} dim={dim}"
        )
        feeds = {"A": matrices[k1], "B": matrices[k2]}
        try:
            sess = engine(onx.SerializeToString(), providers=provider)
        except (NotImplemented, InvalidGraph) as e:
            # not implemented
            errors.append(e)
            continue

        if provider == ["CPUExecutionProvider"]:
            the_feeds = feeds
        else:
            # moving values to CUDA
            device = C_OrtDevice(C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
            try:
                the_feeds = {
                    k: C_OrtValue.ortvalue_from_numpy(v.numpy(), device)
                    for k, v in feeds.items()
                }
            except RuntimeError as e:
                errors.append(f"issue with cuda and type {tt} - {e}")
                continue

        sess._sess.run_with_ort_values(the_feeds, ["C"], None)[0]
        obs = measure_time(
            lambda: sess._sess.run_with_ort_values(the_feeds, ["C"], None)[0],
            repeat=repeat,
            number=number,
        )

    else:
        continue

    obs.update(
        dict(
            engine={"InferenceSession": "ort", "CReferenceEvaluator": "np"}[
                engine.__name__
            ],
            type={
                TensorProto.FLOAT: "f32",
                TensorProto.FLOAT16: "f16",
                TensorProto.INT8: "i8",
                TensorProto.INT16: "i16",
                TensorProto.INT32: "i32",
                TensorProto.UINT32: "u32",
            }[tt],
            M=dim[0],
            N=dim[1],
            K=dim[2],
            cost=numpy.prod(dim) * 4,
            cost_s=f"{numpy.prod(dim) * 4}-{dim[0]}x{dim[1]}x{dim[2]}",
            repeat=repeat,
            number=number,
            provider={"CPUExecutionProvider": "cpu", "CUDAExecutionProvider": "cuda"}[
                provider[0]
            ],
            platform=platform.processor(),
        )
    )
    data.append(obs)
    if unit_test_going() and len(data) >= 2:
        break


df = DataFrame(data)
df.to_excel("plot_bench_gemm.xlsx")
df.to_csv("plot_bench_gemm.csv")
df.drop(["min_exec", "max_exec"], axis=1).to_csv("plot_bench_gemm_.csv")
df

#####################################
# The errors.

for e in list(sorted(set(map(str, errors)))):
    print(e)

##############################################
# Plots
# +++++

piv = pivot_table(
    df, index=["cost"], columns=["engine", "type", "provider"], values="average"
)
piv.reset_index(drop=False).to_excel("plot_bench_gemm_summary.xlsx")
piv.reset_index(drop=False).to_csv("plot_bench_gemm_summary.csv")
print(piv)
piv

########################################
# With the dimensions.
pivs = pivot_table(
    df, index=["cost_s"], columns=["engine", "type", "provider"], values="average"
)
print(pivs)

##############################
# plot

dfi = df[
    df.type.isin({"f32", "f16", "bf16", "f8e4m3", "f8e5m2"}) & df.engine.isin({"ort"})
]
pivi = pivot_table(
    dfi, index=["cost"], columns=["engine", "type", "provider"], values="average"
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
piv.plot(ax=ax[0], title="Gemm performance\nlower is better", logx=True, logy=True)
if pivi.shape[0] > 0:
    pivi.plot(
        ax=ax[1],
        title=f"Gemm performance ORT\n{platform.processor()}",
        logx=True,
        logy=True,
    )
fig.tight_layout()
fig.savefig("plot_bench_gemm.png")
