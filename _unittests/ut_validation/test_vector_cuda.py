import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda_example_py import vector_sum, vector_add
else:
    vector_sum = None
    vector_add = None


class TestVectorCuda(ExtTestCase):
    @unittest.skipIf(vector_add is None, reason="CUDA not available")
    def test_cuda_version(self):
        from onnx_extended import cuda_version

        self.assertTrue(has_cuda())
        self.assertNotEmpty(cuda_version())

    @unittest.skipIf(vector_add is None, reason="CUDA not available")
    def test_vector_add_cuda(self):
        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        t = vector_add(values, values)
        self.assertEqualArray(t, values * 2)

    @unittest.skipIf(vector_add is None, reason="CUDA not available")
    def test_vector_add_cuda_big(self):
        values = numpy.random.randn(3, 224, 224).astype(numpy.float32)
        t = vector_add(values, values)
        self.assertEqualArray(t, values * 2)

    @unittest.skipIf(vector_sum is None, reason="CUDA not available")
    def test_vector_sum_cuda(self):
        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        t = vector_sum(values)
        self.assertEqual(t, values.sum())

    @unittest.skipIf(vector_sum is None, reason="CUDA not available")
    def test_vector_sum_cuda_big(self):
        values = numpy.random.randn(3, 224, 224).astype(numpy.float32)
        t = vector_sum(values)
        self.assertEqual(t, values.sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
