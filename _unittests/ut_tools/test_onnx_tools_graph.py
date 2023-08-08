import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.graph.onnx_graph_struct import Graph


class TestOnnxToolsGraph(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["one"],
                    value=make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
                ),
                make_node("Add", ["one", "one"], ["two"]),
                make_node("Add", ["X", "two"], ["xp"]),
                make_node("MatMul", ["X", "xp"], ["res"]),
                make_node("MatMul", ["X", "res"], ["Z"]),
            ],
            "zoo",
            [X],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    def test_graph_build(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 5)
        cst = []
        for node in graph:
            cst.append(node.is_constant())
        self.assertEqual([True, True, False, False, False], cst)

        ref = CReferenceEvaluator(model)
        x = np.random.random((3, 3)).astype(np.float32)
        z = ref.run(None, dict(X=x))[0]
        self.assertEqual(z.shape, (3, 3))
        self.assertEqualArray(x @ x @ (x + 2), z, atol=1e-5)
        self.assertEqual(len(list(graph)), 5)
        for i in range(0, 5):
            node = graph[i]
            self.assertIn(node.op_type, {"Constant", "Add", "MatMul"})

    def test_graph_replace(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 5)
        indices = graph.replace(2, make_node("Sub", ["one", "one"], ["two"]))
        self.assertEqual(indices, [5])
        self.assertEqual(len(graph), 5)
        self.assertEqual(len(list(graph)), 5)
        ops = []
        for i in range(0, 5):
            if i == 2:
                self.assertRaise(lambda: graph[i], IndexError)
                continue
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["Constant", "Add", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Constant", "Add", "Sub", "MatMul", "MatMul"])
        indices = [node.index for node in graph]
        self.assertEqual(indices, [0, 1, 5, 3, 4])

        graph.simplify()
        self.assertEqual(len(graph), 5)
        self.assertEqual(len(list(graph)), 5)
        ops = []
        for i in range(0, 5):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["Constant", "Add", "Sub", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Constant", "Add", "Sub", "MatMul", "MatMul"])
        indices = [node.index for node in graph]
        self.assertEqual(indices, [0, 1, 5, 3, 4])


if __name__ == "__main__":
    TestOnnxToolsGraph().test_graph_build()
    unittest.main(verbosity=2)
