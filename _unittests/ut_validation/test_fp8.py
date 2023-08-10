import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase

try:
    from onnx_array_api.validation.f8 import search_float32_into_fe4m3
except ImportError:
    # onnx-array-api is not recent enough
    search_float32_into_fe4m3 = None


class TestFloat8(ExtTestCase):
    def test_cast_float32_to_e4m3fn(self):
        from onnx_extended.validation.cython.fp8 import (
            cast_float32_to_e4m3fn,
            cast_e4m3fn_to_float32,
        )

        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        f8 = cast_float32_to_e4m3fn(values)
        back = cast_e4m3fn_to_float32(f8)
        f82 = cast_float32_to_e4m3fn(back)
        self.assertEqualArray(f8, f82)

    @unittest.skipIf(
        search_float32_into_fe4m3 is None, reason="onnx-array-api not recent enough"
    )
    def test_cast_float32_to_e4m3fn_more(self):
        from onnx_extended.validation.cython.fp8 import cast_float32_to_e4m3fn

        vect_search_float32_into_fe4m3 = numpy.vectorize(search_float32_into_fe4m3)

        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        expected = vect_search_float32_into_fe4m3(values).astype(numpy.uint8)
        f8 = cast_float32_to_e4m3fn(values)
        self.assertEqualArray(expected, f8)

        x = numpy.random.randn(4, 4, 4).astype(numpy.float32)
        expected = vect_search_float32_into_fe4m3(x).astype(numpy.uint8)
        f8 = cast_float32_to_e4m3fn(x)
        self.assertEqualArray(expected, f8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
