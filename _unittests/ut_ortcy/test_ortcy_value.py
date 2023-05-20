import unittest
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtCyValue(ExtTestCase):
    def test_api(self):
        from onnx_extended.ortcy.wrap.ortinf import Shape

        shape = Shape()
        shape.set((3, 4))
        self.assertEqual(repr(shape), "Shape((3, 4))")
        self.assertEqual(len(shape), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
