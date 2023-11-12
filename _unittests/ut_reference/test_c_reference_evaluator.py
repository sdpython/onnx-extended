"""
You can run a specific test by using the following syntax.
::

    python _unittest/ut_reference/test_c_reference_evaluator.py
        TestCReferenceEvaluator.test_conv
"""

import os
import unittest
import sys

import numpy as np
from numpy.testing import assert_allclose

import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator

try:
    from onnxruntime import InferenceSession
except ImportError:
    InferenceSession = None
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator


light_model = os.path.join(
    os.path.dirname(onnx.__file__),
    "backend",
    "test",
    "data",
    "light",
    "light_shufflenet.onnx",
)


class TestCReferenceEvaluator(ExtTestCase):
    def conv_test(self, proto_dtype, dtype):
        X = make_tensor_value_info("X", proto_dtype, [None, None, None, None])
        Y = make_tensor_value_info("Y", proto_dtype, [None, None, None, None])
        B = make_tensor_value_info("B", proto_dtype, [None, None, None, None])
        W = make_tensor_value_info("W", proto_dtype, [None, None, None, None])
        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        sess1 = ReferenceEvaluator(onnx_model)
        sess2 = CReferenceEvaluator(onnx_model)

        sH, sW = 5, 6
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=dtype)
                X[0, 0, i, j] = 1.0
                W = np.zeros((1, 1, 3, 3), dtype=dtype)
                W[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 256)

                B = np.array([[[[0]]]], dtype=dtype)
                expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
                got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
                assert_allclose(expected, got)

    @unittest.skipIf(sys.platform == "darwin", reason="crash")
    def test_conv_float(self):
        self.conv_test(TensorProto.FLOAT, np.float32)

    @unittest.skipIf(sys.platform == "darwin", reason="crash")
    def test_conv_double(self):
        self.conv_test(TensorProto.DOUBLE, np.float64)

    @unittest.skipIf(not os.path.exists(light_model), reason="onnx not recent enough")
    @unittest.skipIf(InferenceSession is None, reason="onnxruntime not installed")
    def test_light_model(self):
        sess = CReferenceEvaluator(light_model)
        name = sess.input_names[0]
        shape = [d.dim_value for d in sess.input_types[0].tensor_type.shape.dim]
        img = np.arange(np.prod(shape)).reshape(*shape) / np.prod(shape)
        img = img.astype(np.float32)
        got = sess.run(None, {name: img})
        expected = got[0] * 0 + 1
        expected /= expected.sum().reshape((1, -1))

        self.assertEqual(got[0].shape, (1, 1000))
        self.assertEqual(got[0].dtype, np.float32)
        assert_allclose(expected, got[0], atol=1e-5)

        sess2 = InferenceSession(light_model, providers=["CPUExecutionProvider"])
        got2 = sess2.run(None, {name: img})[0]
        if any(np.isnan(got2.ravel())):
            # There should not be nan values.
            return
        assert_allclose(expected, got2, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
