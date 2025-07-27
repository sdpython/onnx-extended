# source: https://github.com/onnx/onnx/blob/main/onnx/test/helper_test.py
import unittest
from onnx_extended.ext_test_case import ExtTestCase
import onnx_extended.onnx2.helper as oh


class TestOnnx2Helper(ExtTestCase):
    def test_make_operatorsetid(self):
        op = oh.make_operatorsetid("", 19)
        self.assertEqual(op.domain, "")
        self.assertEqual(op.version, 19)
        op = oh.make_operatorsetid("ai.onnx.ml", 5)
        self.assertEqual(op.domain, "ai.onnx.ml")
        self.assertEqual(op.version, 5)
        s = str(op)
        self.assertIn('domain: "ai.onnx.ml",', s)

    def test_make_tensor_type_proto(self) -> None:
        proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        self.assertEqual(proto.tensor_type.elem_type, 2)
        self.assertNotEmpty(proto.tensor_type.shape)
        self.assertEmpty(proto.sequence_type)
        s = str(proto)
        self.assertIn("elem_type: 2,", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
