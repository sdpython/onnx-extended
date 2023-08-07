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
                "level": 0,
                "name": "X",
                "kind": "input",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "Y",
                "kind": "input",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "",
                "kind": "Op",
                "domain": "",
                "type": "Add",
                "inputs": "X,Y",
                "outputs": "z1",
                "input_types": ",",
                "output_types": "FLOAT",
            },
            {
                "name": "z1",
                "kind": "result",
                "level": 0,
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "unk__0xunk__1",
            },
            {
                "level": 0,
                "name": "",
                "kind": "Op",
                "domain": "",
                "type": "Mul",
                "inputs": "X,z1",
                "outputs": "Z",
                "input_types": ",FLOAT",
                "output_types": "FLOAT",
            },
            {
                "name": "Z",
                "kind": "result",
                "level": 0,
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
            },
            {
                "level": 0,
                "name": "Z",
                "kind": "output",
                "type": "tensor",
                "elem_type": "FLOAT",
                "shape": "?x?",
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
