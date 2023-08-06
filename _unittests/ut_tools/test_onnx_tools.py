import unittest
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
from onnx_extended.tools.onnx_tools import enumerate_onnx_node_types


class TestOnnxTools(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["z1"]),
                make_node("Mul", ["X", "z1"], ["Z"]),
            ],
            "add",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model, [
            {
                "elem_type": "FLOAT",
                "kind": "input",
                "level": 0,
                "name": "X",
                "shape": "?x?",
                "type": "tensor",
            },
            {
                "elem_type": "FLOAT",
                "kind": "input",
                "level": 0,
                "name": "Y",
                "shape": "?x?",
                "type": "tensor",
            },
            {"domain": "", "kind": "Op", "level": 0, "name": "", "type": "Add"},
            {
                "elem_type": "FLOAT",
                "kind": "result",
                "name": "z1",
                "shape": "unk__0xunk__1",
                "type": "tensor",
            },
            {"domain": "", "kind": "Op", "level": 0, "name": "", "type": "Mul"},
            {"kind": "result", "name": "Z"},
            {
                "elem_type": "FLOAT",
                "kind": "output",
                "level": 0,
                "name": "Z",
                "shape": "?x?",
                "type": "tensor",
            },
        ]

    def test_enumerate_onnx_node_types(self):
        model, expected = self._get_model()
        res = list(enumerate_onnx_node_types(model))
        self.assertEqual(len(expected), len(res))
        for i, (a, b) in enumerate(zip(expected, res)):
            self.assertEqual(len(a), len(b), msg=f"Item(1) {i} failed a={a}, b={b}.")
            self.assertEqual(set(a), set(b), msg=f"Item(2) {i} failed a={a}, b={b}.")
            for k in sorted(a):
                self.assertEqual(a[k], b[k], msg=f"Item(3) {i} failed a={a}, b={b}.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
