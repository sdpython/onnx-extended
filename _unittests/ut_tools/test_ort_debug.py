import unittest
import numpy as np
from contextlib import redirect_stdout
from io import StringIO
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.tools.ort_debug import enumerate_ort_run


class TestOrtDebug(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.INT64, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"]),
                make_node("Mul", ["X", "z1"], ["z2"]),
                make_node("Cast", ["z2"], ["Z"], to=TensorProto.INT64),
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

    def test_enumerate_ort_run(self):
        model = self._get_model()
        feeds = {
            "X": np.arange(4).reshape((2, 2)).astype(np.float32),
            "Y": np.arange(4).reshape((2, 2)).astype(np.float32),
        }
        expected_names = [["z1"], ["z2"], ["Z"]]
        for i, (names, outs) in enumerate(enumerate_ort_run(model, feeds)):
            self.assertIsInstance(names, list)
            self.assertIsInstance(outs, list)
            self.assertEqual(len(names), len(outs))
            self.assertEqual(names, expected_names[i])

        st = StringIO()
        with redirect_stdout(st):
            for _ in enumerate_ort_run(model, feeds, verbose=2):
                pass
        std = st.getvalue()
        self.assertIn("Add(X, Y) -> z1", std)
        self.assertIn("+ z1: float32(2, 2)", std)
        self.assertIn("Cast(z2, to=7) -> Z", std)

        st = StringIO()
        with redirect_stdout(st):
            for _ in enumerate_ort_run(model, feeds, verbose=3):
                pass
        std = st.getvalue()
        self.assertIn("Add(X, Y) -> z1", std)
        self.assertIn("+ z1: float32(2, 2)", std)
        self.assertIn("[[", std)


if __name__ == "__main__":
    unittest.main(verbosity=2)
