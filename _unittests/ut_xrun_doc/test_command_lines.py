import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._command_lines import _type_shape
from onnx_extended._command_lines_parser import get_main_parser, get_parser_store


class TestCommandLines(ExtTestCase):
    def test_main_parser(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("store", text)

    def test_parser_store(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_store().print_help()
        text = st.getvalue()
        self.assertIn("store", text)
        self.assertIn("verbose", text)

    def test_parse(self):
        checks_str = [
            ("float32(1)", (np.float32, (1,))),
            ("float32(1,N)", (np.float32, (1, "N"))),
            ("float32(1, N)", (np.float32, (1, "N"))),
        ]
        for s_in, expected in checks_str:
            with self.subTest(s_in=s_in):
                dt, shape = _type_shape(s_in)
                self.assertEqual(dt, expected[0])
                self.assertEqual(shape, expected[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
