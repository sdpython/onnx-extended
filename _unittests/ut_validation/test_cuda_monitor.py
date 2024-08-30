import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_monitor import (
        nvml_device_get_count,
        nvml_device_get_memory_info,
        nvml_init,
        nvml_shutdown,
    )
else:
    nvml_init = None


class TestCudaMonitor(ExtTestCase):
    @unittest.skipIf(nvml_init is None, reason="CUDA not available")
    def test_nvml(self):
        nvml_init()
        r = nvml_device_get_count()
        self.assertIsInstance(r, int)
        self.assertGreater(r, 0)
        info = nvml_device_get_memory_info()
        self.assertIsInstance(info, tuple)
        self.assertEqual(len(info), 3)
        self.assertTrue(info[-1] >= max(info[:-1]))
        nvml_shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
