import unittest
import numpy as np
import onnx.numpy_helper as onh
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.cpu._validation import onnx2_read_int64


class TestOnnx2(ExtTestCase):
    def test_onnx2(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        s = a.SerializeToString()
        self.assertEqual(onnx2_read_int64(b"\xac\x02"), (150, 2))
        i = onnx2_read_int64(s[0:])
        self.assertEqual(i, (4, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
