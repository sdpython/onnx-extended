import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation._validation import vector_sum


class TestVectorSum(ExtTestCase):
    def test_vector_sum(self):
        values = [10, 1, 4, 5, 6, 7]
        t1 = vector_sum(1, values, True)
        t2 = vector_sum(1, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

        t1 = vector_sum(2, values, True)
        t2 = vector_sum(2, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

    def test_vector_sum_array(self):
        values = numpy.array([10, 1, 4, 5, 6, 7], dtype=numpy.float32)
        t1 = vector_sum(1, values, True)
        t2 = vector_sum(1, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

        t1 = vector_sum(2, values, True)
        t2 = vector_sum(2, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)


if __name__ == "__main__":
    unittest.main(verbosity=2)
