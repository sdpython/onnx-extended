import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.cpu import (
    __double2float_rn,
    __float2half_rn,
    __half2float,
)


class TestCpuFpEmu(ExtTestCase):
    def test_cast(self):
        self.assertEqual(__double2float_rn(1), 1)
        self.assertEqual(__half2float(__float2half_rn(1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
