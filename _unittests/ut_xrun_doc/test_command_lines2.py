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
from onnxruntime import InferenceSession, SessionOptions
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended._command_lines_parser import (
    get_parser_plot,
    main,
)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
