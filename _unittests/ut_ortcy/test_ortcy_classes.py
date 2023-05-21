import unittest
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtCyClasses(ExtTestCase):
    def test_ort_shape(self):
        from onnx_extended.ortcy.wrap.ortinf import CyOrtShape

        shape = CyOrtShape()
        shape.set((3, 4))
        self.assertEqual(repr(shape), "CyOrtShape((3, 4))")
        self.assertEqual(len(shape), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
