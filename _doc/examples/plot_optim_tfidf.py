"""
.. _l-plot-optim-tfidf:

Measuring performance of TfIdfVectorizer
========================================

The banchmark measures the performance of a TfIdfVectizer along two
parameters, the vocabulary size, the batch size whether. It measures
the benefit of using sparse implementation.

A simple model
++++++++++++++

We start with a model including only one node TfIdfVectorizer.
"""
import itertools
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas
from onnx import ModelProto
from onnx.helper import make_attribute
from tqdm import tqdm
from onnxruntime import InferenceSession, SessionOptions
from onnx_extended.ext_test_case import measure_time, unit_test_going
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ortops.optim.cpu import get_ort_ext_libs


def make_onnx(n_words: int) -> ModelProto:
    from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
    from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

    # from onnx_array_api.light_api import start
    # onx = (
    #     start(opset=19, opsets={"ai.onnx.ml": 3})
    #     .vin("X", elem_type=TensorProto.INT64)
    #     .ai.onnx.TfIdfVectorizer(
    #     ...
    #     )
    #     .rename(Y)
    #     .vout(elem_type=TensorProto.FLOAT)
    #     .to_onnx()
    # )
    onx = OnnxTfIdfVectorizer(
        "X",
        mode="TF",
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        ngram_counts=[0],
        ngram_indexes=np.arange(n_words).tolist(),
        pool_int64s=np.arange(n_words).tolist(),
        output_names=["Y"],
    ).to_onnx(inputs=[("X", Int64TensorType())], outputs=[("Y", FloatTensorType())])
    #     .rename(Y)
    #     .vout(elem_type=TensorProto.FLOAT)
    #     .to_onnx()
    # )
    return onx


onx = make_onnx(7)
ref = CReferenceEvaluator(onx)
got = ref.run(None, {"X": np.array([[0, 1], [2, 3]], dtype=np.int64)})
print(got)

#################################
# It works as expected. Let's now compare the execution
# with onnxruntime for different batch size and vocabulary size.
#
# Benchmark
# +++++++++


def make_sessions(
    onx: ModelProto,
) -> Tuple[InferenceSession, InferenceSession, InferenceSession]:
    # first: onnxruntime
    ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])

    # second: custom kernel equivalent to the onnxruntime implementation
    for node in onx.graph.node:
        if node.op_type == "TfIdfVectorizer":
            node.domain = "onnx_extented.ortops.optim.cpu"
            # new_add = make_attribute("sparse", 1)
            # node.attribute.append(new_add)

    d = onx.opset_import.add()
    d.domain = "onnx_extented.ortops.optim.cpu"
    d.version = 1

    r = get_ort_ext_libs()
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    cus = InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    # third: with sparse
    for node in onx.graph.node:
        if node.op_type == "TfIdfVectorizer":
            new_add = make_attribute("sparse", 1)
            node.attribute.append(new_add)
    cussp = InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    return ref, cus, cussp


if unit_test_going():
    vocabulary_sizes = [10, 20]
    batch_sizes = [10, 20]
else:
    vocabulary_sizes = [100, 1000, 5000, 10000]
    batch_sizes = [500, 1000, 2000]
confs = list(itertools.product(vocabulary_sizes, batch_sizes))

data = []
for voc_size, batch_size in tqdm(confs):
    onx = make_onnx(voc_size)
    ref, cus, sparse = make_sessions(onx)

    feeds = dict(
        X=(np.arange(batch_size * 10) % voc_size)
        .reshape((batch_size, -1))
        .astype(np.int64)
    )

    # reference
    ref.run(None, feeds)
    obs = measure_time(lambda: ref.run(None, feeds), max_time=1)
    obs["name"] = "ref"
    obs.update(dict(voc_size=voc_size, batch_size=batch_size))
    data.append(obs)

    # custom
    cus.run(None, feeds)
    obs = measure_time(lambda: cus.run(None, feeds), max_time=1)
    obs["name"] = "custom"
    obs.update(dict(voc_size=voc_size, batch_size=batch_size))
    data.append(obs)

    # sparse
    sparse.run(None, feeds)
    obs = measure_time(lambda: sparse.run(None, feeds), max_time=1)
    obs["name"] = "sparse"
    obs.update(dict(voc_size=voc_size, batch_size=batch_size))
    data.append(obs)

df = pandas.DataFrame(data)
df.to_csv("plot_optim_tdidf.csv", index=False)
print(df.head())


#####################################
# Plots
# +++++

piv = pandas.pivot_table(
    df, index=["voc_size", "name"], columns="batch_size", values="average"
)
print(piv)

fig, ax = plt.subplots(1, 1)
piv.plot(ax=ax, logy=True, logx=True)
fig.savefig("plot_optim_tfidf.png")
