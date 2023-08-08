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
        self.assertEqual(len(graph), 6)
        cst = []
        for node in graph:
            cst.append(node.is_constant())
        self.assertEqual([False, True, True, False, False, False], cst)

        ref = CReferenceEvaluator(model)
        x = np.random.random((3, 3)).astype(np.float32)
        z = ref.run(None, dict(X=x))[0]
        self.assertEqual(z.shape, (3, 3))
        self.assertEqualArray(x @ x @ (x + 2), z, atol=1e-5)
        self.assertEqual(len(list(graph)), 6)
        for i in range(0, 6):
            node = graph[i]
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul"})
        text = str(graph)
        tn = str(graph[1])
        self.assertEqual(tn, "Node(1, <parent>, <Constant>) [] -> [one]")
        self.assertEqual(text, "Graph(...) [X] -> [Z]")

    def test_graph_build_initializer(self):
        onnx_model = make_model(
            make_graph(
                [make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])],
                "graph",
                [make_tensor_value_info("x", TensorProto.FLOAT, (None, None, None))],
                [make_tensor_value_info("y", TensorProto.FLOAT, (1, 6, 2))],
                initializer=[
                    make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                    make_tensor("ends", TensorProto.INT64, (2,), (2, 2)),
                    make_tensor("axes", TensorProto.INT64, (2,), (0, 2)),
                ],
            )
        )
        check_model(onnx_model)
        graph = Graph(onnx_model)
        self.assertEqual(len(graph), 5)
        for node in graph:
            self.assertEqual("Node(0, <parent>, <input>) [] -> [x]", str(node))
            break
        self.assertEqual("Graph(...) [x] -> [y]", str(graph))

    def test_graph_opsets(self):
        model = self._get_model()
        graph = Graph(model)
        opsets = graph.get_opsets()
        main = graph.get_opset()
        self.assertEqual(opsets[""], main)

    def test_graph_replace(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 6)
        indices = graph.replace(3, make_node("Sub", ["X", "two"], ["xp"]))
        self.assertEqual(indices, [6])
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            if i == 3:
                self.assertRaise(lambda: graph[i], IndexError)
                continue
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual(indices, [0, 1, 2, 6, 4, 5])

        graph.simplify(False)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)

        graph.simplify(True)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)

    def test_graph_remove(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 6)
        graph.replace(3, make_node("Sub", ["X", "X"], ["xp"]))
        graph.simplify(False)
        removed = graph.remove_unused_nodes()
        self.assertEqual(len(removed), 2)
        self.assertEqual(str(removed[0]), "Node(2, <parent>, <Add>) [one,one] -> [two]")
        self.assertEqual(str(removed[1]), "Node(1, <parent>, <Constant>) [] -> [one]")

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
