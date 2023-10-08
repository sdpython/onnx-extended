import os
import sys
from onnx_extended import __version__ as extversion, check_installation
from onnx_extended.ext_test_case import ExtTestCase

try:
    import tomllib as toml

    fmt = "rb"
except ImportError:
    import toml

    fmt = "r"
import unittest


class TestVersion(ExtTestCase):
    def test_version_toml(self):
        this = os.path.dirname(__file__)
        name = os.path.join(this, "..", "..", "pyproject.toml")
        with open(name, fmt) as f:
            tom = toml.load(f)
        self.assertEqual("onnx-extended", tom["project"]["name"])
        self.assertEqual(extversion, tom["project"]["version"])

    def test_check_installation(self):
        # It seems using both ortops and ortcy and customops lead to
        # munmap_chunk(): invalid pointer on Linux
        check_installation(val=True, ortcy=True, ortops=sys.platform != "inux")


if __name__ == "__main__":
    # TestCommandLines().test_command_external()
    unittest.main(verbosity=2)
