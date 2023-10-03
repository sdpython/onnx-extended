import unittest
import numpy
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_function,
    make_model,
    make_graph,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.tools.onnx_nodes import (
    select_model_inputs_outputs,
    onnx_remove_node_unused,
)


class TestOptimOnnxUnused(ExtTestCase):
    def test_onnx_remove_unused(self):
        dtype = numpy.float32
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Add", ["X", "init2"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("Mul", ["X1", "inter"], ["Xm"]),
                    make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                [
                    from_array(numpy.array([1], dtype=dtype), name="init1"),
                    from_array(numpy.array([2], dtype=dtype), name="init2"),
                    from_array(numpy.array([3], dtype=dtype), name="init3"),
                ],
            )
        )

        check_model(model_def0)
        model_def = select_model_inputs_outputs(
            model_def0, "inter", remove_unused=False
        )
        check_model(model_def)
        new_model = onnx_remove_node_unused(model_def)
        n0, i0 = len(model_def0.graph.node), len(model_def0.graph.initializer)
        n2, i2 = len(model_def.graph.node), len(model_def.graph.initializer)
        n3, i3 = len(new_model.graph.node), len(new_model.graph.initializer)
        self.assertEqual((n0, i0), (5, 3))
        self.assertEqual((n2, i2), (1, 3))
        self.assertEqual((n3, i3), (1, 1))

    def test_onnx_remove_unused_function(self):
        dtype = numpy.float32
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Add", ["X", "init2"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("fct", ["X1", "inter"], ["Xm"], domain="local"),
                    make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                [
                    from_array(numpy.array([1], dtype=dtype), name="init1"),
                    from_array(numpy.array([2], dtype=dtype), name="init2"),
                    from_array(numpy.array([3], dtype=dtype), name="init3"),
                ],
            ),
            functions=[
                make_function(
                    "local",
                    "fct",
                    ["x", "y"],
                    ["z"],
                    [
                        make_node("Div", ["x", "y"], ["z"]),
                    ],
                    opset_imports=[make_opsetid("", 18)],
                )
            ],
            opset_imports=[make_opsetid("", 18), make_opsetid("local", 1)],
        )

        check_model(model_def0)
        model_def = select_model_inputs_outputs(
            model_def0, "inter", remove_unused=False
        )
        check_model(model_def)
        new_model = onnx_remove_node_unused(model_def)
        n0, i0 = len(model_def0.graph.node), len(model_def0.graph.initializer)
        n2, i2 = len(model_def.graph.node), len(model_def.graph.initializer)
        n3, i3 = len(new_model.graph.node), len(new_model.graph.initializer)
        self.assertEqual((n0, i0), (5, 3))
        self.assertEqual((n2, i2), (1, 3))
        self.assertEqual((n3, i3), (1, 1))
        self.assertEqual(len(model_def0.functions), 1)
        self.assertEqual(len(model_def.functions), 1)
        self.assertEqual(
            len(new_model.functions), 1
        )  # should be zero if unused function are removed.


if __name__ == "__main__":
    unittest.main()
