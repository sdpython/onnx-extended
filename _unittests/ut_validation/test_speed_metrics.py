import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.validation._validation import benchmark_cache, benchmark_cache_tree


class TestSpeedMetrics(ExtTestCase):
    def test_benchmark_cache(self):
        res = benchmark_cache(1000, False)
        self.assertGreater(res, 0)

    def test_benchmark_cache_tree(self):
        res = benchmark_cache_tree(1000)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1000)
        last = res[-1]
        self.assertEqual(last.trial, 0)
        self.assertEqual(last.row, 999)

        res = benchmark_cache_tree(2000)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2000)


if __name__ == "__main__":
    # TestSpeedMetrics().test_benchmark_cache_tree()
    unittest.main(verbosity=2)
