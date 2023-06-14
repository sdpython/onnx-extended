import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import gemm_test
else:
    gemm_test = None


class TestCudaGemm(ExtTestCase):
    @unittest.skipIf(gemm_test is None, reason="CUDA not available")
    def test_gemm_test_float32(self):
        gemm_test(0)

    @unittest.skipIf(gemm_test is None, reason="CUDA not available")
    def test_gemm_test_float8(self):
        gemm_test(1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
