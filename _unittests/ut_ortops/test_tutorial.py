import unittest
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnxruntime import InferenceSession, SessionOptions
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtOpTutorial(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    def test_my_custom_ops(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "MyCustomOp", ["X", "A"], ["Y"], domain="onnx_extented.ortops.tutorial.cpu"
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("onnx_extented.ortops.tutorial.cpu", 1)]
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(onnx_model.SerializeToString(), opts)
        a = numpy.random.randn(2, 2).astype(numpy.float32)
        b = numpy.random.randn(2, 2).astype(numpy.float32)
        feeds = {"X": a, "A": b}
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(a + b, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
