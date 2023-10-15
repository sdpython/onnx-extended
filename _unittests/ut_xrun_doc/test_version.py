import os
import sys
from onnx_extended import __version__ as extversion, check_installation
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._config import ORT_VERSION, ORT_VERSION_INT
from onnx_extended import ort_version, ort_version_int

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

    def test_ort_version(self):
        self.assertEqual(ort_version(), ORT_VERSION)
        v = (
            ORT_VERSION_INT // 1000,
            (ORT_VERSION_INT % 1000) // 10,
            ORT_VERSION_INT % 10,
        )
        self.assertEqual(ort_version_int(), v)


if __name__ == "__main__":
    # TestCommandLines().test_command_external()
    unittest.main(verbosity=2)
