# coding: utf-8
import unittest
from logging import getLogger
import packaging.version as pv
import numpy
import onnx
from sklearn.feature_extraction.text import CountVectorizer
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings
from onnx_extended.reference import CReferenceEvaluator

TARGET_OPSET = 18


class TestCTfIdefVectorizer(ExtTestCase):
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
        oinf = CReferenceEvaluator(onx)
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
        oinf = CReferenceEvaluator(onx)
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
        oinf = CReferenceEvaluator(onx)
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
        oinf = CReferenceEvaluator(onx)
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
        oinf = CReferenceEvaluator(onx)
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
        oinf = CReferenceEvaluator(onx)
        got = oinf.run(None, {"X": corpus})
        self.assertEqualArray(exp.todense().astype(numpy.float32), got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
