import unittest
from onnx_extended.ext_test_case import ExtTestCase


class TestSpeedMetrics(ExtTestCase):
    def test_benchmark_cache(self):
        from onnx_extended.validation.cpu._validation import (
            benchmark_cache,
        )

        res = benchmark_cache(1000, False)
        self.assertGreater(res, 0)

    def test_benchmark_cache_tree(self):
        from onnx_extended.validation.cpu._validation import (
            benchmark_cache_tree,
        )

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
