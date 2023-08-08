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
from onnx_extended.tools.graph.onnx_graph_transformer import quantize_float8


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
        text = str(graph)
        tn = str(graph[0])
        self.assertEqual(tn, "Node(0, <parent>, <Constant>) [] -> [one]")
        self.assertEqual(text, "Graph(...)")

    def test_graph_replace(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 5)
        indices = graph.replace(2, make_node("Sub", ["X", "two"], ["xp"]))
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

        graph.simplify(False)
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
        self.assertEqual([0, 1, 2, 3, 4], indices)

        graph.simplify(True)
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
        self.assertEqual([0, 1, 2, 3, 4], indices)

    def test_graph_remove(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 5)
        graph.replace(2, make_node("Sub", ["X", "X"], ["xp"]))
        graph.simplify(False)
        removed = graph.remove_unused_nodes()
        self.assertEqual(len(removed), 2)
        self.assertEqual(str(removed[0]), "Node(1, <parent>, <Add>) [one,one] -> [two]")
        self.assertEqual(str(removed[1]), "Node(0, <parent>, <Constant>) [] -> [one]")

    def _get_model_32(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["mat"],
                    value=make_tensor(
                        "one",
                        TensorProto.FLOAT,
                        [3, 2],
                        list(float(i) for i in range(1, 7)),
                    ),
                ),
                make_node("MatMul", ["X", "mat"], ["Z"]),
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

    def test_quantize_f8(self):
        model = self._get_model_32()
        graph = Graph(model)
        new_graph = quantize_float8(graph)
        self.assertGreater(len(new_graph), len(graph))


if __name__ == "__main__":
    unittest.main(verbosity=2)
