import unittest
import pandas
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.plotting.data import hhistograms_data, vhistograms_data
from onnx_extended.plotting.benchmark import hhistograms, vhistograms


class TestCReferenceEvaluator(ExtTestCase):
    def test_plotting_hhistograms(self):
        import matplotlib.pyplot as plt

        plt.clf()
        df = pandas.DataFrame(hhistograms_data())
        ax = hhistograms(df)
        self.assertNotEmpty(ax)

    def test_plotting_vhistograms(self):
        import matplotlib.pyplot as plt

        plt.clf()
        df = pandas.DataFrame(vhistograms_data())
        ax = vhistograms(df)
        self.assertNotEmpty(ax)


if __name__ == "__main__":
    unittest.main(verbosity=2)
