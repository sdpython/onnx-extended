import unittest
from contextlib import redirect_stdout
from io import StringIO
from onnx_extended.ext_test_case import ExtTestCase
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
