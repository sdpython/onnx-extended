import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._command_lines import _type_shape
from onnx_extended._command_lines_parser import get_main_parser, get_parser_store, main


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
            ("float32(1)", (np.float32, (1,), None)),
            ("float32(1,N)", (np.float32, (1, "N"), None)),
            ("float32(1, N)", (np.float32, (1, "N"), None)),
            ("(1, N)", (None, (1, "N"), None)),
            ("float32", (np.float32, None, None)),
            ("float32(1,N):U10", (np.float32, (1, "N"), "U10")),
            ("float32:U10", (np.float32, None, "U10")),
            ("(1,N):U10", (None, (1, "N"), "U10")),
            (":U10", (None, None, "U10")),
        ]
        for s_in, expected in checks_str:
            with self.subTest(s_in=s_in):
                dt, shape, law = _type_shape(s_in)
                self.assertEqual(dt, expected[0])
                self.assertEqual(shape, expected[1])
                self.assertEqual(law, expected[2])

    def test_parser(self):
        args = [
            "store",
            "-m",
            "model",
            "-o",
            "output",
            "-i",
            "input1.pb",
            "-i",
            "input2.pb",
            "-v",
        ]
        parser = get_main_parser()
        res = parser.parse_args(args[:1])
        self.assertEqual(res.cmd, "store")
        parser = get_parser_store()
        res = parser.parse_args(args[1:])
        self.assertEqual(res.model, "model")
        self.assertEqual(res.out, "output")
        self.assertEqual(res.verbose, True)
        self.assertEqual(res.runtime, "CReferenceEvaluator")
        self.assertEqual(res.input, ["input1.pb", "input2.pb"])

    def test_command_store(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [5, 6])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["res"]),
                make_node("Cos", ["res"], ["Z"]),
            ],
            "g",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])

        with tempfile.TemporaryDirectory() as root:
            model_file = os.path.join(root, "model.onnx")
            with open(model_file, "wb") as f:
                f.write(onnx_model.SerializeToString())
            args = [
                "store",
                "-m",
                model_file,
                "-o",
                root,
                "-i",
                "float32(5,6):U10",
                "-i",
                "U10",
            ]
            main(args)

            ds = list(sorted(os.listdir(root)))
            self.assertEqual(
                ds,
                list(sorted(["test_node_0_Add", "test_node_1_Cos", "model.onnx"])),
            )
            self.assertExists(model_file)
            for sub in ds[1:]:
                fols = os.listdir(os.path.join(root, sub))
                self.assertEqual(
                    list(sorted(fols)), list(sorted(["test_data_set_0", "model.onnx"]))
                )

    def test_command_display(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [5, 6])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node("Add", ["X", "Y"], ["res"]),
                make_node("Cos", ["res"], ["Z"]),
            ],
            "g",
            [X, Y],
            [Z],
        )
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        st = StringIO()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            f.write(onnx_model.SerializeToString())
            f.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".csv") as root:
                with redirect_stdout(st):
                    args = ["display", "-m", f.name, "-s", root.name]
                    main(args)
        text = st.getvalue()
        self.assertIn("input     tensor    X         FLOAT     ?x?", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
