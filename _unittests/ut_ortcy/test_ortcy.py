import unittest
import os
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


class TestOrtCy(ExtTestCase):
    def test_add(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "add", [X, Y], [Z])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        data = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data):
            os.mkdir(data)
        name = os.path.join(data, "add.onnx")
        if not os.path.exists(name):
            with open(name, "wb") as f:
                f.write(onnx_model.SerializeToString())
        self.assertExists(name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
