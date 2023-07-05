import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import (
        gemm_benchmark_test,
        get_device_prop,
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
        self.assertIn("GB", r["name"])

    def gemm_test(self, test):
        r = gemm_benchmark_test(test)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 19)
        self.assertEqual(r["N"], 5)

    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float32(self):
        for i in range(0, 5):
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
