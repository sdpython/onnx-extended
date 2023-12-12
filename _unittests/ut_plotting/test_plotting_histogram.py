import unittest
import pandas
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.plotting.data import histograms_data
from onnx_extended.plotting.benchmark import vhistograms


class TestCReferenceEvaluator(ExtTestCase):
    def test_plotting_histograms(self):
        import matplotlib.pyplot as plt

        plt.clf()
        df = pandas.DataFrame(histograms_data())
        ax = vhistograms(df)
        self.assertNotEmpty(ax)


if __name__ == "__main__":
    unittest.main(verbosity=2)
