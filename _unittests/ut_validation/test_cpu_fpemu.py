import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.cpu._validation import (
    double2float_rn,
    float2half_rn,
    half2float,
)


class TestCpuFpEmu(ExtTestCase):
    def test_cast(self):
        self.assertEqual(double2float_rn(1), 1)
        self.assertEqual(half2float(float2half_rn(1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
