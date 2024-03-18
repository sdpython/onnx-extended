# coding: utf-8
import unittest
from logging import getLogger
import packaging.version as pv
import numpy
import onnx
from onnx.reference import ReferenceEvaluator
from sklearn.feature_extraction.text import CountVectorizer
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings


TARGET_OPSET = 18


def make_ort_session(onx):
    from onnxruntime import InferenceSession, SessionOptions
    from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

    sess_check = InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    for node in onx.graph.node:
        if node.op_type == "TfIdfVectorizer":
            node.domain = "onnx_extented.ortops.optim.cpu"

    d = onx.opset_import.add()
    d.domain = "onnx_extented.ortops.optim.cpu"
    d.version = 1

    r = get_ort_ext_libs()
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    sess = InferenceSession(
        onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    return sess_check, sess


class TestTfIdfVectorizer(ExtTestCase):
    def setUp(self):
        logger = getLogger("skl2onnx")
        logger.disabled = True

    def test_onnx_tfidf_vectorizer(self):
        from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

        inputi = numpy.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(
            numpy.int64
        )
        output = numpy.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        ).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            numpy.int64
        )  # bigrams

        op = OnnxTfIdfVectorizer(
            "tokens",
            op_version=TARGET_OPSET,
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
            output_names=["out"],
        )
        onx = op.to_onnx(
            inputs=[("tokens", Int64TensorType())], outputs=[("out", FloatTensorType())]
        )

        check, oinf = ReferenceEvaluator(onx), CReferenceEvaluator(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

        check, oinf = make_ort_session(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    def test_onnx_tfidf_vectorizer_skip5(self):
        from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

        inputi = numpy.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(
            numpy.int64
        )
        output = numpy.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        ).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            numpy.int64
        )  # bigrams

        op = OnnxTfIdfVectorizer(
            "tokens",
            op_version=TARGET_OPSET,
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
            output_names=["out"],
        )
        onx = op.to_onnx(
            inputs=[("tokens", Int64TensorType())], outputs=[("out", FloatTensorType())]
        )
        check, oinf = make_ort_session(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    def test_onnx_tfidf_vectorizer_unibi_skip5(self):
        from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

        inputi = numpy.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(
            numpy.int64
        )
        output = numpy.array(
            [[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]]
        ).astype(numpy.float32)

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            numpy.int64
        )  # bigrams

        op = OnnxTfIdfVectorizer(
            "tokens",
            op_version=TARGET_OPSET,
            mode="TF",
            min_gram_length=1,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
            output_names=["out"],
        )
        onx = op.to_onnx(
            inputs=[("tokens", Int64TensorType())], outputs=[("out", FloatTensorType())]
        )
        check, oinf = make_ort_session(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    def test_onnx_tfidf_vectorizer_bi_skip0(self):
        from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

        inputi = numpy.array([[1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]).astype(
            numpy.float32
        )

        ngram_counts = numpy.array([0, 4]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2, 3, 4, 5, 6]).astype(numpy.int64)
        pool_int64s = numpy.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            numpy.int64
        )  # bigrams

        op = OnnxTfIdfVectorizer(
            "tokens",
            op_version=TARGET_OPSET,
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
            output_names=["out"],
        )
        onx = op.to_onnx(
            inputs=[("tokens", Int64TensorType())], outputs=[("out", FloatTensorType())]
        )
        check, oinf = make_ort_session(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    def test_onnx_tfidf_vectorizer_empty(self):
        from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
        from skl2onnx.algebra.onnx_ops import OnnxTfIdfVectorizer

        inputi = numpy.array([[1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]]).astype(numpy.int64)
        output = numpy.array([[1.0, 1.0, 1.0]]).astype(numpy.float32)

        ngram_counts = numpy.array([0, 0]).astype(numpy.int64)
        ngram_indexes = numpy.array([0, 1, 2]).astype(numpy.int64)
        pool_int64s = numpy.array([5, 6, 7, 8, 6, 7]).astype(  # unigrams
            numpy.int64
        )  # bigrams

        op = OnnxTfIdfVectorizer(
            "tokens",
            op_version=TARGET_OPSET,
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
            output_names=["out"],
        )
        onx = op.to_onnx(
            inputs=[("tokens", Int64TensorType())], outputs=[("out", FloatTensorType())]
        )
        check, oinf = make_ort_session(onx)
        res = check.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())
        res = oinf.run(None, {"tokens": inputi})
        self.assertEqual(output.tolist(), res[0].tolist())

    @ignore_warnings(UserWarning)
    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="onnx not recent enough",
    )
    def test_sklearn_count_vectorizer(self):
        from skl2onnx import to_onnx

        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        )

        vect = CountVectorizer()
        vect.fit(corpus)
        exp = vect.transform(corpus)
        onx = to_onnx(vect, corpus, target_opset=TARGET_OPSET)
        try:
            check, oinf = make_ort_session(onx)
        except Exception as e:
            if "type inference failed" in str(e):
                # Type inference failed
                # see https://github.com/microsoft/onnxruntime/pull/17497
                raise unittest.SkipTest("does not work for onnxruntime<1.17.1")
            if "This is an invalid model." in str(e):
                raise unittest.SkipTest("not yet implemented for strings")
            raise e
        got = check.run(None, {"X": corpus})
        self.assertEqualArray(exp.todense().astype(numpy.float32), got[0])
        got = oinf.run(None, {"X": corpus})
        self.assertEqualArray(exp.todense().astype(numpy.float32), got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
