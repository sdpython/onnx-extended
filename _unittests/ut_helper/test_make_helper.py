import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.helper import (
    make_dynamic_quantize_linear_function_proto,
    make_reshape_transpose_function_proto,
    make_reshape_transpose_back_function_proto,
)
from onnx_extended.reference import CReferenceEvaluator


class TestMakeHelper(ExtTestCase):
    def test_dynamic_quantize_linear(self):
        onx = make_model(
            make_graph(
                [
                    make_node(
                        "DynamicQuantizeLinear",
                        ["X"],
                        ["q", "scale", "zp"],
                        to=TensorProto.FLOAT8E4M3FN,
                        domain="qtest",
                    ),
                    make_node("Cast", ["q"], ["Y"], to=TensorProto.FLOAT),
                ],
                "name",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [
                    make_tensor_value_info("Y", TensorProto.FLOAT, [None]),
                    make_tensor_value_info("scale", TensorProto.FLOAT, [0]),
                ],
            ),
            functions=[
                make_dynamic_quantize_linear_function_proto(domain="qtest", opset=18)
            ],
            opset_imports=[
                make_opsetid("", 18),
                make_opsetid("qtest", 1),
            ],
        )
        ref = CReferenceEvaluator(onx)
        feeds = {"X": np.array([1, 4, 5, 10, -10], dtype=np.float32)}
        got = ref.run(None, feeds)
        self.assertEqualArray(
            np.array([14.0, 56.0, 72.0, 144.0, -144.0], dtype=np.float32), got[0]
        )
        self.assertEqualArray(np.array(0.06952997, dtype=np.float32), got[1])

        ref = CReferenceEvaluator(onx)
        feeds = {"X": np.array([1, 4, np.nan, 10, -10], dtype=np.float32)}
        got = ref.run(None, feeds)
        self.assertEqualArray(
            np.array([14.0, 56.0, np.nan, 128.0, -128.0], dtype=np.float32), got[0]
        )
        self.assertEqualArray(np.array(0.073612, dtype=np.float32), got[1], atol=1e-5)

    def test_reshape_transpose(self):
        onx = make_model(
            make_graph(
                [
                    make_node(
                        "ReshapeTranspose0",
                        ["X"],
                        ["Y"],
                        domain="qtest",
                    ),
                ],
                "name",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
            ),
            functions=[
                make_reshape_transpose_function_proto(domain="qtest", opset=18, index=0)
            ],
            opset_imports=[
                make_opsetid("", 18),
                make_opsetid("qtest", 1),
            ],
        )
        ref = CReferenceEvaluator(onx)
        feeds = {"X": np.arange(24).reshape((2, 3, 4)).astype(np.float32)}
        got = ref.run(None, feeds)
        self.assertEqualArray(
            np.arange(24).reshape((2, 3, 4)).reshape((-1, 4)).T.astype(np.float32),
            got[0],
        )

    def test_reshape_transpose_back(self):
        onx = make_model(
            make_graph(
                [
                    make_node(
                        "ReshapeTransposeBack0",
                        ["X", "shape"],
                        ["Y"],
                        domain="qtest",
                    ),
                ],
                "name",
                [
                    make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                    make_tensor_value_info("shape", TensorProto.INT64, [None]),
                ],
                [make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])],
            ),
            functions=[
                make_reshape_transpose_back_function_proto(
                    domain="qtest", opset=18, index=0
                )
            ],
            opset_imports=[
                make_opsetid("", 18),
                make_opsetid("qtest", 1),
            ],
        )
        ref = CReferenceEvaluator(onx)
        feeds = {
            "X": np.arange(24).reshape((-1, 4)).astype(np.float32),
            "shape": np.array([2, 3, 4], dtype=np.int64),
        }
        got = ref.run(None, feeds)
        self.assertEqualArray(
            np.arange(24).reshape((2, 3, 4)).astype(np.float32),
            got[0],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
