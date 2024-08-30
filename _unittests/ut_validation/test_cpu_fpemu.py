import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.cpu._validation import has_sse3


class TestCpuFpEmu(ExtTestCase):
    @unittest.skipIf(not has_sse3(), "SSE3 not available")
    def test_cast(self):
        from onnx_extended.validation.cpu._validation import (
            double2float_rn,
            float2half_rn,
            half2float,
        )

        self.assertEqual(double2float_rn(1), 1)
        self.assertEqual(half2float(float2half_rn(1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
