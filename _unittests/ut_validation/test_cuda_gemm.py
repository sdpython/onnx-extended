import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import (
        gemm_benchmark_test,
        get_device_prop,
        cuda_device_count,
        cuda_device_memory,
        cuda_devices_memory,
    )
else:
    gemm_benchmark_test = None
    get_device_prop = None


class TestCudaGemm(ExtTestCase):
    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_get_device_prop(self):
        r = get_device_prop()
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 12)

    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_cuda_device_count(self):
        r = cuda_device_count()
        self.assertIsInstance(r, int)
        self.assertGreater(r, 0)

    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_cuda_device_memory(self):
        r = cuda_device_memory(0)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)

    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_cuda_devices_memory(self):
        r = cuda_devices_memory()
        n = cuda_device_count()
        self.assertIsInstance(r, list)
        self.assertEqual(len(r), n)
        self.assertIsInstance(r[0], tuple)
        self.assertEqual(len(r[0]), 2)

    def gemm_test(self, test):
        r = gemm_benchmark_test(test)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 24)
        self.assertEqual(r["N"], 10)

    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float32(self):
        for i in range(5):
            with self.subTest(test=i):
                self.gemm_test(i)

    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float8(self):
        r = get_device_prop()
        if r["major"] < 9:
            return
        for i in range(5, 15):
            if i in {8, 9, 10, 12, 13}:
                # still invalid
                continue
            with self.subTest(test=i):
                self.gemm_test(i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
