import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation._validation import (
    vector_add,
    vector_sum,
    vector_sum_array,
    vector_sum_array_parallel,
    vector_sum_array_avx,
    vector_sum_array_avx_parallel,
)


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
        t1 = vector_sum_array(1, values, True)
        t2 = vector_sum_array(1, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

        t1 = vector_sum_array(2, values, True)
        t2 = vector_sum_array(2, values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

    def test_vector_sum_array_parallel(self):
        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_parallel(16, values, True)
        t2 = vector_sum_array_parallel(16, values, False)
        self.assertEqual(t, t1)
        self.assertEqual(t, t2)

    def test_vector_sum_array_avx(self):
        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_avx(16, values)
        self.assertEqual(t, t1)

    def test_vector_sum_array_avx_parallel(self):
        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_avx_parallel(16, values)
        self.assertEqual(t, t1)

    def test_vector_add_exc(self):
        # This test checks function vector_add
        # raises an exception if the dimension do not match.
        v1 = numpy.ones((3, 4), dtype=numpy.float32)
        v2 = numpy.ones((4,), dtype=numpy.float32)
        self.assertRaise(lambda: vector_add(v1, v2), RuntimeError)
        v2 = numpy.ones((4, 3), dtype=numpy.float32)
        self.assertRaise(lambda: vector_add(v1, v2), RuntimeError)

    def test_vector_add(self):
        v1 = numpy.ones((3, 4), dtype=numpy.float32)
        v2 = numpy.ones((3, 4), dtype=numpy.float32)
        v3 = vector_add(v1, v2)
        self.assertEqual(v3.shape, (3, 4))
        # Adds some code to check the addition is ok.


if __name__ == "__main__":
    unittest.main(verbosity=2)
