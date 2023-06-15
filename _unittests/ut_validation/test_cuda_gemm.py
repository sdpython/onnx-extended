import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import gemm_benchmark_test
else:
    gemm_test = None


class TestCudaGemm(ExtTestCase):
    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float32(self):
        r = gemm_benchmark_test(0)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 18)
        self.assertEqual(r["N"], 5)
        if __name__ == "__main__":
            import pprint

            pprint.pprint(r)

    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float8(self):
        r = gemm_benchmark_test(1)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 18)
        self.assertEqual(r["N"], 5)
        if __name__ == "__main__":
            import pprint

            pprint.pprint(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
