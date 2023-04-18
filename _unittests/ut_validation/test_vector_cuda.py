import unittest
import numpy
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda, compiled_with_cuda, cuda_version

if has_cuda():
    from onnx_extended.validation.cuda_example_py import vector_sum, vector_add
else:
    vector_sum = None
    vector_add = None


class TestVectorCuda(ExtTestCase):
    def test_cuda_version(self):
        if vector_sum is not None:
            self.assertTrue(has_cuda())
            self.assertNotEmpty(cuda_version())
        else:
            self.assertFalse(has_cuda())

    def test_compiled_with_cuda(self):
        if vector_sum is not None:
            self.assertTrue(compiled_with_cuda())
        else:
            self.assertFalse(compiled_with_cuda())

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

    @unittest.skipIf(vector_add is None, reason="CUDA not available")
    def test_vector_add_cuda_bigger(self):
        values = numpy.random.randn(30, 224, 224).astype(numpy.float32)
        t = vector_add(values, values)
        self.assertEqualArray(t, values * 2)

    @unittest.skipIf(vector_sum is None, reason="CUDA not available")
    def test_vector_sum_cuda(self):
        values = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.float32)
        t = vector_sum(values)
        self.assertEqual(t, values.sum().astype(numpy.float32))

    @unittest.skipIf(vector_sum is None, reason="CUDA not available")
    def test_vector_sum_cuda_big(self):
        values = numpy.random.randn(3, 224, 224).astype(numpy.float32)
        t = vector_sum(values)
        self.assertAlmostEqual(t, values.sum().astype(numpy.float32), rtol=1e-5)

    @unittest.skipIf(vector_sum is None, reason="CUDA not available")
    def test_vector_sum_cuda_bigger(self):
        values = numpy.random.randn(30, 224, 224).astype(numpy.float32)
        t = vector_sum(values)
        self.assertAlmostEqual(t, values.sum().astype(numpy.float32), rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
