import unittest
import numpy as np
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda


class TestCudaFpemu(ExtTestCase):
    @unittest.skipIf(not has_cuda(), reason="CUDA not available")
    def test_fpemu_cuda_forward(self):
        from onnx_extended.validation.cuda.cuda_example_py import (
            fpemu_cuda_forward,
        )

        values = np.array([-2, -1, 0, 1, 2, 3], dtype=np.float32)
        res = fpemu_cuda_forward(values)
        print(res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
