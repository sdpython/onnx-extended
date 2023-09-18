import unittest
import os
import tempfile
import numpy as np
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
from onnx_extended.tools.test_onnx import save_for_benchmark_or_test


class TestTestOnnx(ExtTestCase):
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

    def test_save_for_benchmark_or_test(self):
        model = self._get_model()
        inputs = [
            np.arange(4).reshape((2, 2)).astype(np.float32),
            np.arange(4).reshape((2, 2)).astype(np.float32),
        ]

        with tempfile.TemporaryDirectory() as temp:
            save_for_benchmark_or_test(temp, "t1", model, inputs)
            self.assertExists(os.path.join(temp, "t1"))
            self.assertExists(os.path.join(temp, "t1", "model.onnx"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
