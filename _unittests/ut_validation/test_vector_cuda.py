import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation.cuda_example_py import vector_sum


class TestVectorCuda(ExtTestCase):
    def test_vector_cuda(self):
        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        t = vector_sum(values)
        self.assertEqual(t, values.sum())

    def test_vector_cuda_big(self):
        values = numpy.random.randn(3, 224, 224).astype(numpy.float32)
        t = vector_sum(values)
        self.assertEqual(t, values.sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
