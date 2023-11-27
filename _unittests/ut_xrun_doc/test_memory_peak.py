import os
import time
import numpy as np
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.memory_peak import get_memory_rss, start_spying_on
import unittest


class TestMemoryPeak(ExtTestCase):
    def test_memory(self):
        mem = get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    def test_spy(self):
        p = start_spying_on()
        res = []
        for i in range(0, 10):
            time.sleep(0.005)
            res.append(np.empty(i * 1000000))
        del res
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertLessEqual(pres["end"], pres["max_peak"])
        self.assertLessEqual(pres["begin"], pres["max_peak"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
