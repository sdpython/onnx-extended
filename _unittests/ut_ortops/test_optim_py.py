import unittest
import numpy
from onnx import TensorProto
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
)
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.ortops.optim.optimize import change_onnx_operator_domain


class TestOrtOpOptimPy(ExtTestCase):
    def test_replace_add(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(onnx_model, op_type="Add", new_op_type="Sub")
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "Sub")
        ref = ReferenceEvaluator(repl)
        x = numpy.arange(5).astype(numpy.float32)
        y = (x * 10).astype(numpy.float32)
        got = ref.run(None, {"X": x, "Y": y})
        self.assertEqualArray(x - y, got[0])

    def test_replace_argmin_1(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("ArgMin", ["X"], ["Z"], axis=0)
        graph = make_graph([node], "g", [X], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model, op_type="ArgMin", new_op_type="ArgMin", axis=None
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "ArgMin")
        self.assertEqual(len(repl.graph.node[0].attribute), 0)
        ref = ReferenceEvaluator(repl)
        x = numpy.arange(5).astype(numpy.float32)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(numpy.argmin(x).reshape((-1,)), got[0])

    def test_replace_argmin_2(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("ArgMin", ["X"], ["Z"])
        graph = make_graph([node], "g", [X], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model, op_type="ArgMin", new_op_type="ArgMin", axis=0
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "ArgMin")
        self.assertEqual(len(repl.graph.node[0].attribute), 1)
        ref = ReferenceEvaluator(repl)
        x = numpy.arange(5).astype(numpy.float32)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(numpy.argmin(x).reshape((-1,)), got[0])

    def test_replace_argmin_3(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        node = make_node("ArgMin", ["X"], ["Z"], axis=1)
        graph = make_graph([node], "g", [X], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model, op_type="ArgMin", new_op_type="ArgMax", axis=0
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "ArgMax")
        self.assertEqual(len(repl.graph.node[0].attribute), 1)
        ref = ReferenceEvaluator(repl)
        x = numpy.arange(4).astype(numpy.float32).reshape((2, -1))
        got = ref.run(None, {"X": x})
        self.assertEqualArray(numpy.argmax(x, axis=0, keepdims=1), got[0])

    def test_replace_domain(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model,
            op_type="Add",
            new_op_type="Sub",
            new_op_domain="NEW",
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "Sub")
        self.assertIn('domain: "NEW"', str(repl))

    def test_replace_domain_att(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model,
            op_type="Add",
            new_op_type="Sub",
            new_op_domain="NEW",
            ATTR=6,
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "Sub")
        self.assertIn('domain: "NEW"', str(repl))
        self.assertIn('name: "ATTR"', str(repl))
        self.assertIn("i: 6", str(repl))

    def test_replace_domain_att_same(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        check_model(onnx_model)

        repl = change_onnx_operator_domain(
            onnx_model,
            op_type="Add",
            new_op_domain="NEW",
            ATTR=6,
        )
        check_model(repl)
        self.assertEqual(len(repl.graph.node), 1)
        self.assertEqual(repl.graph.node[0].op_type, "Add")
        self.assertIn('domain: "NEW"', str(repl))
        self.assertIn('name: "ATTR"', str(repl))
        self.assertIn("i: 6", str(repl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
