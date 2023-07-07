import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase


class TestVectorSum(ExtTestCase):
    def test_vector_sum_c(self):
        from onnx_extended.validation.cython.vector_function_cy import vector_sum_c

        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        t1 = vector_sum_c(values, True)
        t2 = vector_sum_c(values, False)
        self.assertEqual(t1, 33)
        self.assertEqual(t2, 33)

    def test_vector_sum(self):
        from onnx_extended.validation.cpu._validation import vector_sum

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
        from onnx_extended.validation.cpu._validation import vector_sum_array

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
        from onnx_extended.validation.cpu._validation import vector_sum_array_parallel

        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_parallel(16, values, True)
        t2 = vector_sum_array_parallel(16, values, False)
        self.assertEqual(t, t1)
        self.assertEqual(t, t2)

    def test_vector_sum_array_avx(self):
        from onnx_extended.validation.cpu._validation import vector_sum_array_avx

        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_avx(16, values)
        self.assertEqual(t, t1)

    def test_vector_sum_array_avx_parallel(self):
        from onnx_extended.validation.cpu._validation import (
            vector_sum_array_avx_parallel,
        )

        values = numpy.arange(16 * 16).reshape((-1, 16)).astype(numpy.float32)
        t = values.sum()
        t1 = vector_sum_array_avx_parallel(16, values)
        self.assertEqual(t, t1)

    def test_vector_add_exc(self):
        from onnx_extended.validation.cpu._validation import vector_add

        # This test checks function vector_add
        # raises an exception if the dimension do not match.
        v1 = numpy.ones((3, 4), dtype=numpy.float32)
        v2 = numpy.ones((4,), dtype=numpy.float32)
        self.assertRaise(lambda: vector_add(v1, v2), RuntimeError)
        v2 = numpy.ones((4, 3), dtype=numpy.float32)
        self.assertRaise(lambda: vector_add(v1, v2), RuntimeError)

    def test_vector_add(self):
        from onnx_extended.validation.cpu._validation import vector_add

        v1 = numpy.ones((3, 4), dtype=numpy.float32)
        v2 = (numpy.ones((3, 4)) * 10).astype(numpy.float32)
        v3 = vector_add(v1, v2)
        self.assertEqual(v3.shape, (3, 4))
        self.assertEqualArray(v1 + v2, v3)

    def test_vector_add_c(self):
        from onnx_extended.validation.cython.vector_function_cy import vector_add_c

        t1 = numpy.arange(10).reshape((2, 5)).astype(numpy.float32)
        t2 = numpy.arange(10).reshape((2, 5)).astype(numpy.float32)
        res = t1 + t2
        got = vector_add_c(t1, t2)
        self.assertEqualArray(res, got)
        t0 = numpy.array([[0]], dtype=numpy.float32)
        self.assertRaise(lambda: vector_add_c(t0, t1), ValueError)
        t0 = numpy.array([0], dtype=numpy.float32)
        self.assertRaise(lambda: vector_add_c(t0, t1), ValueError)
        t0 = numpy.array([0], dtype=numpy.float32)
        t1i = t1.astype(numpy.int32)
        self.assertRaise(lambda: vector_add_c(t1i, t2), TypeError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
