import unittest
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtOpTutorial(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
