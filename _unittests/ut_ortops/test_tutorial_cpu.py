import unittest
import numpy
from onnx import TensorProto
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnx_extended.ortops.tutorial.cpu import documentation
from onnx_extended.reference import CReferenceEvaluator

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtOpTutorialCpu(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 3)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
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
            graph,
            opset_imports=[make_opsetid("onnx_extented.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        a = numpy.random.randn(2, 2).astype(numpy.float32)
        b = numpy.random.randn(2, 2).astype(numpy.float32)
        feeds = {"X": a, "A": b}
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(a + b, got)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_my_custom_ops_with_attributes(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.DOUBLE, [None, None])
        A = make_tensor_value_info("A", TensorProto.DOUBLE, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.DOUBLE, [None, None])
        node1 = make_node(
            "MyCustomOpWithAttributes",
            ["X", "A"],
            ["Y"],
            domain="onnx_extented.ortops.tutorial.cpu",
            att_string="string_att",
            att_int64=5,
            att_float=4.5,
            att_tensor=from_array(numpy.array([[5.1]], dtype=numpy.float64)),
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[make_opsetid("onnx_extented.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        a = numpy.random.randn(2, 2).astype(numpy.float64)
        b = numpy.random.randn(2, 2).astype(numpy.float64)
        feeds = {"X": a, "A": b}
        cst = 5.1 + 4.5 + 5 + ord("s")
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(a + b + cst, got)

    def _get_dql_model(self, domain, opset):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        scale = make_tensor_value_info("scale", TensorProto.FLOAT, [])
        onnx_model = make_model(
            make_graph(
                [
                    make_node(
                        "DynamicQuantizeLinear",
                        ["X"],
                        ["yu", "scale", "zp"],
                        domain=domain,
                        to=TensorProto.FLOAT8E4M3FN,
                    ),
                    make_node(
                        "Cast",
                        ["yu"],
                        ["Y"],
                        to=TensorProto.FLOAT,
                    ),
                ],
                "test",
                [X],
                [Y, scale],
            ),
            opset_imports=[make_opsetid(domain, opset)]
            if domain == ""
            else [make_opsetid(domain, opset), make_opsetid("", 19)],
            ir_version=9,
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(InferenceSession is None, reason="onnxruntime not installed")
    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_dynamic_quantize_linear(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        x = numpy.array(
            [
                [-1.9386303424835205, -0.927788257598877, -0.4964291751384735],
                [-0.7981147170066833, 0.5894935131072998, -0.5586161017417908],
            ],
            dtype=numpy.float32,
        )
        # expected = numpy.array([[244, 235, 228], [234, 103, 230]], dtype=float8e4m3fn)
        expected = numpy.array([[-192, -88, -48], [-80, 60, -56]], dtype=numpy.float32)
        expected_scale = numpy.array(0.010128305, dtype=numpy.float32)
        feeds = {"X": x}

        ref = CReferenceEvaluator(self._get_dql_model("", 20))
        got, scale = ref.run(None, feeds)
        self.assertEqualArray(expected_scale, scale)
        self.assertEqualArray(expected, got)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            self._get_dql_model(
                "onnx_extented.ortops.tutorial.cpu", 1
            ).SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
        got, scale = sess.run(None, feeds)
        self.assertEqualArray(expected_scale, scale)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    TestOrtOpTutorialCpu().test_dynamic_quantize_linear()
    unittest.main(verbosity=2)
