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


if __name__ == "__main__":
    unittest.main(verbosity=2)
