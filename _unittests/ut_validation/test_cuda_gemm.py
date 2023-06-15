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
    def test_get_device_prop(self):
        r = get_device_prop()
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 12)
        if __name__ == "__main__":
            import pprint

            pprint.pprint(r)

    def gemm_test(self, test):
        r = gemm_benchmark_test(test)
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 18)
        self.assertEqual(r["N"], 5)
        if __name__ == "__main__":
            import pprint

            pprint.pprint(r)

    @unittest.skipIf(gemm_benchmark_test is None, reason="CUDA not available")
    def test_gemm_test_float32(self):
        r = get_device_prop()
        for i in range(0, 12):
            if r["major"] <= 6 and i >= 5:
                # float 8 not supported
                break
            with self.subTest(test=i):
                self.gemm_test(i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
