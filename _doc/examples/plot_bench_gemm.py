"""
.. _l-example-bench-gemm:

Measuring performance about Gemm
================================

Differents types, differents backend, differents

Onnx Model
++++++++++
"""
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
from onnxruntime import InferenceSession, get_available_providers
from onnxruntime.capi._pybind_state import (
    OrtValue as C_OrtValue,
    OrtDevice as C_OrtDevice,
)
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented, InvalidGraph
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ext_test_case import unit_test_going, measure_time


def create_model(mat_type=TensorProto.FLOAT):
    A = make_tensor_value_info("A", mat_type, [None, None])
    B = make_tensor_value_info("B", mat_type, [None, None])
    C = make_tensor_value_info("C", mat_type, [None, None])
    node1 = make_node("MatMul", ["A", "B"], ["C"])
    graph = make_graph([node1], "a", [A, B], [C])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)], ir_version=9)
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
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)], ir_version=9)
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
providers = [["CPUExecutionProvider"], ["CUDAExecutionProvider"]]
# M, N, K
dims = [
    (10, 10, 10),
    (61, 62, 63),
    (64, 64, 64),
    (65, 66, 67),
    (100, 100, 100),
    (128, 128, 128),
    (256, 256, 256),
    (400, 400, 400),
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
            vect = (numpy.random.randn(i, j) * 10).astype(numpy.float32)
            ov = to_ort_value(vect)
            sess = InferenceSession(create_cast(tt).SerializeToString())
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
        repeat, number = 50, 50
    elif max(dim) <= 256:
        repeat, number = 25, 25
    else:
        repeat, number = 10, 10

    onx = create_model(tt)
    k1 = (tt, dim[0], dim[2])
    k2 = (tt, dim[2], dim[1])
    assert k1 in matrices
    assert k2 in matrices

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
            sess._sess.run_with_ort_values(feeds, ["C"], None)[0]
            obs = measure_time(
                lambda: sess._sess.run_with_ort_values(feeds, ["C"], None)[0],
                repeat=repeat,
                number=number,
            )
        else:
            continue

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
            cost=numpy.prod(dim),
            repeat=repeat,
            number=number,
            provider=provider[0][:4],
        )
    )
    data.append(obs)
    if unit_test_going() and len(data) >= 2:
        break


df = DataFrame(data)
print(df)

#####################################
# The errors.

for e in errors:
    print(e)

##############################################
# Plots
# +++++

piv = pivot_table(df, index=["cost"], columns=["engine", "type"], values="average")
print(piv)


fig, ax = plt.subplots(1, 1, figsize=(12, 4))
piv.plot(ax=ax, title="Gemm performance", logx=True, logy=True)
fig.savefig("plot_bench_gemm.png")
