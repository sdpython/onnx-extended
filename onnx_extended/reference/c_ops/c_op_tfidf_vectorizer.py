from typing import Any, Dict
import numpy as np
from onnx import NodeProto
from onnx.reference.op_run import OpRun
from .cpu.c_op_tfidf_vectorizer_py_ import PyRuntimeTfIdfVectorizer


class TfIdfVectorizer(OpRun):
    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema=schema)
        self.rt_ = None

    def _init(self, **kwargs):
        self.rt_ = PyRuntimeTfIdfVectorizer()

        if kwargs["pool_strings"] is not None and len(kwargs["pool_strings"]) > 0:
            self.pool_strings_ = kwargs["pool_strings"]
            mapping = {}
            pool_int64s = []
            for i, w in enumerate(kwargs["pool_strings"]):
                if w not in mapping:
                    # 1-gram are processed first.
                    mapping[w] = i
                pool_int64s.append(mapping[w])
            self.mapping_ = mapping
        else:
            self.mapping_ = None
            self.pool_strings_ = None
            pool_int64s = None

        self.rt_.init(
            kwargs["max_gram_length"],
            kwargs["max_skip_count"],
            kwargs["min_gram_length"],
            kwargs["mode"],
            kwargs["ngram_counts"],
            kwargs["ngram_indexes"],
            kwargs["pool_int64s"] or pool_int64s,
            kwargs["weights"] or [],
            False,
        )

    def _run(
        self,
        x,
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ):
        if self.rt_ is None:
            self._init(
                max_gram_length=max_gram_length,
                max_skip_count=max_skip_count,
                min_gram_length=min_gram_length,
                mode=mode,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s,
                pool_strings=pool_strings,
                weights=weights,
            )

        if self.mapping_ is not None:
            xi = np.empty(x.shape, dtype=np.int64)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    try:
                        xi[i, j] = self.mapping_[x[i, j]]
                    except KeyError:
                        xi[i, j] = -1
            res = self.rt_.compute(xi)
        else:
            res = self.rt_.compute(x)
        if len(x.shape) > 1:
            return (res.reshape((x.shape[0], -1)),)
        return (res,)
