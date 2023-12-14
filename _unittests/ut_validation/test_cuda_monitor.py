import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_monitor import (
        get_device_prop,
        cuda_device_count,
        cuda_device_memory,
        cuda_devices_memory,
        nvml_device_get_memory_info,
        nvml_init,
        nvml_shutdown,
    )
else:
    get_device_prop = None


class TestCudaMonitor(ExtTestCase):
    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_get_device_prop(self):
        r = get_device_prop()
        self.assertIsInstance(r, dict)
        self.assertEqual(len(r), 12)
        self.assertIn("NVIDIA", r["name"])

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

    @unittest.skipIf(get_device_prop is None, reason="CUDA not available")
    def test_nvml(self):
        nvml_init()
        info = nvml_device_get_memory_info()
        self.assertIsInstance(info, tuple)
        self.assertEqual(len(info), 3)
        print(info)
        self.assertTrue(info[-1] >= max(info[:-1]))
        nvml_shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
