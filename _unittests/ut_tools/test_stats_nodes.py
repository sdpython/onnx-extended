import os
import unittest
import numpy
from onnx import GraphProto, NodeProto, TensorProto
from onnx.checker import check_model
from onnx.numpy_helper import from_array
import onnx.backend.test as test_data
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.tools.stats_nodes import (
    enumerate_nodes,
    enumerate_stats_nodes,
    HistStatistics,
    HistTreeStatistics,
    TreeStatistics,
)
from onnx_extended.tools import load_model


class TestStatsNodes(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"], name="A"),
                make_node("Mul", ["X", "z1"], ["Z"], name="M"),
            ],
            "add",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    def test_enumerate_nodes(self):
        model = self._get_model()
        res = list(enumerate_nodes(model))
        names = [i[0] for i in res]
        self.assertEqual([("add", "A"), ("add", "M")], names)

    def test_enumerate_nodes_loop(self):
        model_path = os.path.join(
            os.path.dirname(test_data.__file__),
            "data",
            "node",
            "test_loop16_seq_none",
            "model.onnx",
        )
        model = load_model(model_path)
        res = list(enumerate_nodes(model))
        names = [i[0] for i in res]
        expected = [
            ("test_loop16_seq_none", "#0"),
            ("test_loop16_seq_none", "#0/body", "#0"),
            ("test_loop16_seq_none", "#0/body", "#1"),
            ("test_loop16_seq_none", "#0/body", "#2"),
            ("test_loop16_seq_none", "#0/body", "#3"),
            ("test_loop16_seq_none", "#0/body", "#3/else_branch", "#0"),
            ("test_loop16_seq_none", "#0/body", "#3/then_branch", "#0"),
            ("test_loop16_seq_none", "#0/body", "#3/then_branch", "#1"),
            ("test_loop16_seq_none", "#0/body", "#4"),
            ("test_loop16_seq_none", "#0/body", "#5"),
            ("test_loop16_seq_none", "#0/body", "#6"),
            ("test_loop16_seq_none", "#0/body", "#7"),
            ("test_loop16_seq_none", "#0/body", "#8"),
            ("test_loop16_seq_none", "#0/body", "#9"),
            ("test_loop16_seq_none", "#0/body", "#10"),
            ("test_loop16_seq_none", "#0/body", "#11"),
        ]
        self.assertEqual(expected, names)

    def test_stats_nodes(self):
        from skl2onnx import to_onnx

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(3, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        onx = to_onnx(rf, X[:1])

        stats = list(enumerate_stats_nodes(onx))
        # self.assertEqual(len(stats), 1)
        for name, node, stat in stats:
            self.assertIsInstance(name, tuple)
            self.assertIsInstance(node, GraphProto)
            self.assertIsInstance(stat.node, NodeProto)
            self.assertEqual(stat["max_featureid"], 1)
            self.assertEqual(stat["n_features"], 2)
            self.assertEqual(stat["n_outputs"], 1)
            self.assertEqual(stat["n_rules"], 2)
            self.assertEqual(stat["n_trees"], 3)
            self.assertIsInstance(stat["trees"], list)
            self.assertIsInstance(stat["trees"][0], TreeStatistics)
            self.assertIsInstance(stat["features"], list)
            self.assertIsInstance(stat["features"][0], HistTreeStatistics)

    def _get_model_init(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"], name="A"),
                make_node("Mul", ["X", "z1"], ["Z"], name="M"),
            ],
            "add",
            [X],
            [Z],
            [from_array(numpy.array([[2, 3], [4, 5]], dtype=numpy.float32), name="Y")],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    def test_stats_inits_nodes(self):
        onx = self._get_model_init()

        stats = list(enumerate_stats_nodes(onx))
        # self.assertEqual(len(stats), 1)
        n = 0
        for name, parent, stat in stats:
            self.assertEqual(("add", "Y"), name)
            self.assertIsInstance(parent, GraphProto)
            self.assertIsInstance(stat, HistStatistics)
            self.assertEqual(stat["shape"], (2, 2))
            self.assertEqual(stat["dtype"], numpy.float32)
            self.assertEqual(stat["sparse"], 0)
            n += 1
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
