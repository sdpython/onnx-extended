import unittest
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.args import get_parsed_args


class TestArgs(ExtTestCase):
    def test_get_parsed_args(self):
        parsed = get_parsed_args(
            "ut",
            args=["-s", "large"],
            expose="scenario",
            scenarios={"large": "large model"},
        )
        self.assertEqual(parsed.scenario, "large")

    def test_get_parsed_args_exp(self):
        parsed = get_parsed_args(
            "ut", args=["-s", "large"], expose="", scenarios={"large": "large model"}
        )
        self.assertEqual(parsed.scenario, "large")

    def test_get_parsed_args_a(self):
        parsed = get_parsed_args("ut", args=["--ppp", "5"], ppp=("j", "zoo"))
        self.assertEqual(parsed.ppp, "5")

    def test_get_parsed_args_x(self):
        parsed = get_parsed_args(
            "ut", args=["-r", "5"], ppp=("j", "zoo"), expose="number,repeat"
        )
        self.assertEqual(parsed.ppp, "j")
        self.assertEqual(parsed.repeat, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
