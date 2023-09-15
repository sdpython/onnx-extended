# coding: utf-8
import unittest
import numpy
from sklearn.feature_extraction import FeatureHasher
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings


class TestHash(ExtTestCase):
    def test_hash(self):
        from onnx_extended.validation.cpu._validation import (
            murmurhash3_bytes_s32_ort,
            murmurhash3_bytes_s32_sklearn,
        )

        values = ["a", "b", "d", "abd", "é"]
        for v in values:
            with self.subTest(v=v):
                o = murmurhash3_bytes_s32_ort(v)
                s = murmurhash3_bytes_s32_sklearn(v)
                self.assertEqual(o, s)

    @ignore_warnings(DeprecationWarning)
    def test_feature_hasher_int64(self):
        from onnxruntime import InferenceSession
        from skl2onnx import to_onnx

        data = numpy.array(
            [
                "a",
                "b",
                "d",
                "abd",
                "6-11yrs",
                "6-11yrs",
                "11-15yrs",
            ]
        ).reshape((-1, 1))
        fe = FeatureHasher(
            n_features=16,
            input_type="string",
            alternate_sign=False,
            dtype=numpy.int64,
        )
        fe.fit(data)
        expected = fe.transform(data).todense()
        onx = to_onnx(fe, data)

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": data})
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_feature_hasher_float32(self):
        from onnxruntime import InferenceSession
        from skl2onnx import to_onnx

        data = numpy.array(
            [
                "a",
                "b",
                "d",
                "abd",
                "6-11yrs",
                "6-11yrs",
                "11-15yrs",
            ]
        ).reshape((-1, 1))
        fe = FeatureHasher(
            n_features=16,
            input_type="string",
            alternate_sign=True,
            dtype=numpy.float32,
        )
        fe.fit(data)
        expected = fe.transform(data).todense()
        onx = to_onnx(fe, data)

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": data})
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    import logging

    for name in ["onnx-extended", "skl2onnx"]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    unittest.main(verbosity=2)
