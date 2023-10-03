import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.parser import parse_model
from onnxruntime import InferenceSession, SessionOptions
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._command_lines_parser import (
    get_parser_merge,
    get_parser_plot,
    main,
)
from onnx_extended.tools.onnx_io import save_model, load_model


class TestCommandLines2(ExtTestCase):
    def test_parser_plot(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_plot().print_help()
        text = st.getvalue()
        self.assertIn("kind", text)
        self.assertIn("verbose", text)

    def test_command_store(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        graph = make_graph(
            [
                make_node("Neg", ["X"], ["res"]),
                make_node("Cos", ["res"], ["Z"]),
            ],
            "g",
            [X],
            [Z],
        )
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        check_model(onnx_model)

        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        with tempfile.TemporaryDirectory() as root:
            csv = os.path.join(root, "o.csv")
            png = os.path.join(root, "o.png")
            args = [
                "plot",
                "-i",
                prof,
                "-k",
                "profile_node",
                "-c",
                csv,
                "-o",
                png,
                "-v",
            ]
            st = StringIO()
            with redirect_stdout(st):
                main(args)
            self.assertIn("[plot_profile] save", st.getvalue())
            self.assertExists(png)
            self.assertExists(csv)

        os.remove(prof)

    def test_parser_merge(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_merge().print_help()
        text = st.getvalue()
        self.assertIn("output", text)

    def test_merge(self):
        M1_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10, "com.microsoft": 1]
            >
            agraph (float[N, M] A0, float[N, M] A1, float[N, M] _A
                    ) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
            {
                B00 = Add(A0, A1)
                B10 = Sub(A0, A1)
                B20 = Mul(A0, A1)
            }
            """

        M2_DEF = """
            <
                ir_version: 7,
                opset_import: [ "": 10, "com.microsoft": 1]
            >
            agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21
                    ) => (float[N, M] D0)
            {
                C0 = Add(B01, B11)
                C1 = Sub(B11, B21)
                M1 = Mul(C0, C1)
            }
            """

        m1 = parse_model(M1_DEF)
        m2 = parse_model(M2_DEF)

        with tempfile.TemporaryDirectory() as root:
            name1 = os.path.join(root, "m1.onnx")
            name2 = os.path.join(root, "m2.onnx")
            name3 = os.path.join(root, "m3.onnx")
            save_model(m1, name1)
            save_model(m2, name2)
            args = [
                "merge",
                "--m1",
                name1,
                "--m2",
                name2,
                "-o",
                name3,
                "-m",
                "B00,B01;B10,B11;B20,B21",
            ]
            st = StringIO()
            with redirect_stdout(st):
                main(args)
            text = st.getvalue()
            self.assertEmpty(text)
            m3 = load_model(name3)
            self.assertNotEmpty(m3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
