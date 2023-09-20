import os
from onnx_extended import __version__ as extversion
from onnx_extended.ext_test_case import ExtTestCase

try:
    import tomllib as toml
except ImportError:
    import toml
import unittest


class TestVersion(ExtTestCase):
    def test_version_toml(self):
        this = os.path.dirname(__file__)
        name = os.path.join(this, "..", "..", "pyproject.toml")
        with open(name, "r") as f:
            tom = toml.load(f)
        self.assertEqual("onnx-extended", tom["project"]["name"])
        self.assertEqual(extversion, tom["project"]["version"])


if __name__ == "__main__":
    # TestCommandLines().test_command_external()
    unittest.main(verbosity=2)
