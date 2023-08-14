import os
import tempfile
import unittest
import sys
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from onnx import TensorProto, load
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
)
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._command_lines import _type_shape
from onnx_extended._command_lines_parser import (
    get_main_parser,
    get_parser_store,
    get_parser_display,
    get_parser_print,
    get_parser_quantize,
    main,
)


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

    def test_parser_display(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_display().print_help()
        text = st.getvalue()
        self.assertIn("display", text)

    def test_parser_print(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_print().print_help()
        text = st.getvalue()
        self.assertIn("print", text)

    def test_parser_quantize(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_quantize().print_help()
        text = st.getvalue()
        self.assertIn("quantize", text)

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

    @unittest.skipIf(sys.platform == "win32", reason="permision issue")
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
        self.assertIn("input       tensor      X           FLOAT", text)

    def test_command_print_exc1(self):
        self.assertRaise(
            lambda: main(["print", "-i", "__any__.onnx"]), FileNotFoundError
        )
        self.assertRaise(lambda: main(["print", "-i", __file__]), ValueError)

    @unittest.skipIf(sys.platform == "win32", reason="permision issue")
    def test_command_print_exc2(self):
        with tempfile.NamedTemporaryFile(suffix=".pb") as f:
            f.write(b"Rrrrrrrrrrrrrr")
            f.seek(0)
            self.assertRaise(lambda: main(["print", "-i", f.name]), RuntimeError)

    @unittest.skipIf(sys.platform == "win32", reason="permision issue")
    def test_command_print_model(self):
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
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            f.write(onnx_model.SerializeToString())
            f.seek(0)

            st = StringIO()
            with redirect_stdout(st):
                args = ["print", "-i", f.name]
                main(args)
            text = st.getvalue()
            self.assertIn("Type:", text)
            self.assertIn('op_type: "Cos"', text)

    @unittest.skipIf(sys.platform == "win32", reason="permision issue")
    def test_command_print_tensor(self):
        tensor = make_tensor("dummy", TensorProto.FLOAT8E4M3FN, [4], [0, 1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix=".pb") as f:
            f.write(tensor.SerializeToString())
            f.seek(0)

            st = StringIO()
            with redirect_stdout(st):
                args = ["print", "-i", f.name]
                main(args)
            text = st.getvalue()
            self.assertIn("Type:", text)
            self.assertIn("17", text)

    def _get_model_32(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["mat"],
                    value=make_tensor(
                        "one",
                        TensorProto.FLOAT,
                        [3, 2],
                        list(float(i) for i in range(11, 17)),
                    ),
                ),
                make_node("MatMul", ["X", "mat"], ["Z"]),
            ],
            "zoo",
            [X],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_command_quantize_model(self):
        onnx_model = self._get_model_32()
        with tempfile.TemporaryDirectory() as fold:
            model_file = os.path.join(fold, "model.onnx")
            with open(model_file, "wb") as f:
                f.write(onnx_model.SerializeToString())
            model_out = os.path.join(fold, "out.onnx")

            st = StringIO()
            with redirect_stdout(st):
                args = ["quantize", "-i", model_file, "-o", model_out, "-k", "fp8"]
                main(args)
            text = st.getvalue()
            self.assertEqual(text, "")
            self.assertExists(model_out)
            with open(model_out, "rb") as f:
                content = load(f)
            types = [n.op_type for n in content.graph.node]
            self.assertEqual(
                types,
                [
                    "Transpose",
                    "DynamicQuantizeLinear",
                    "Constant",
                    "Constant",
                    "GemmFloat8",
                ],
            )

    def test_command_quantize_model_local(self):
        onnx_model = self._get_model_32()
        with tempfile.TemporaryDirectory() as fold:
            model_file = os.path.join(fold, "model.onnx")
            with open(model_file, "wb") as f:
                f.write(onnx_model.SerializeToString())
            model_out = os.path.join(fold, "out.onnx")

            st = StringIO()
            with redirect_stdout(st):
                args = [
                    "quantize",
                    "-i",
                    model_file,
                    "-o",
                    model_out,
                    "-k",
                    "fp8",
                    "-l",
                ]
                main(args)
            text = st.getvalue()
            self.assertEqual(text, "")
            self.assertExists(model_out)
            with open(model_out, "rb") as f:
                content = load(f)
            types = [n.op_type for n in content.graph.node]
            self.assertEqual(
                types,
                [
                    "Transpose",
                    "DynamicQuantizeLinear",
                    "Constant",
                    "Constant",
                    "GemmFloat8",
                ],
            )
            types = set(n.domain for n in content.graph.node)
            self.assertEqual(
                types,
                {"", "com.microsoft", "local.quant.domain"},
            )


if __name__ == "__main__":
    TestCommandLines().test_command_quantize_model()
    unittest.main(verbosity=2)
