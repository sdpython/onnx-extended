import os
import unittest
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_graph,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnxruntime import InferenceSession, SessionOptions
import matplotlib.pyplot as plt
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings
from onnx_extended.tools.js_profile import (
    js_profile_to_dataframe,
    plot_ort_profile,
    plot_ort_profile_timeline,
    _process_shape,
)


class TestJsProfile(ExtTestCase):
    def test_shapes(self):
        tests = [
            (
                "U8[1x128x768]+F+U8",
                [{"uint8": [1, 128, 768]}, {"float": []}, {"uint8": []}],
            ),
            ("F[1x128x768]", [{"float": [1, 128, 768]}]),
        ]
        for expected, shapes in tests:
            with self.subTest(shapes=shapes):
                out = _process_shape(shapes)
                self.assertEqual(expected, out)

    def _get_model(self):
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Abs", ["X"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("Mul", ["X1", "inter"], ["Xm"]),
                    make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                [
                    from_array(np.array([1], dtype=np.float32), name="init1"),
                    from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model_def0)
        return model_def0

    def test_js_profile_to_dataframe(self):
        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)
        self.assertEqual(df.shape, (189, 18))
        self.assertEqual(
            set(df.columns),
            set(
                [
                    "cat",
                    "pid",
                    "tid",
                    "dur",
                    "ts",
                    "ph",
                    "name",
                    "args_op_name",
                    "op_name",
                    "args_thread_scheduling_stats",
                    "args_output_size",
                    "args_parameter_size",
                    "args_activation_size",
                    "args_node_index",
                    "args_provider",
                    "event_name",
                    "iteration",
                    "it==0",
                ]
            ),
        )

        df = js_profile_to_dataframe(prof, agg=True)
        self.assertEqual(df.shape, (17, 1))
        self.assertEqual(list(df.columns), ["dur"])

        df = js_profile_to_dataframe(prof, agg_op_name=True)
        self.assertEqual(df.shape, (189, 17))
        self.assertEqual(
            set(df.columns),
            set(
                [
                    "cat",
                    "pid",
                    "tid",
                    "dur",
                    "ts",
                    "ph",
                    "name",
                    "args_op_name",
                    "op_name",
                    "args_thread_scheduling_stats",
                    "args_output_size",
                    "args_parameter_size",
                    "args_activation_size",
                    "args_node_index",
                    "args_provider",
                    "event_name",
                    "iteration",
                ]
            ),
        )

        os.remove(prof)

    @ignore_warnings(UserWarning)
    def test_plot_profile_2(self):
        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        # fig.savefig("graph1.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    @ignore_warnings(UserWarning)
    def test_plot_profile_2_shape(self):
        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True, with_shape=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        # fig.savefig("graph1.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    @ignore_warnings(UserWarning)
    def test_plot_profile_agg(self):
        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True, agg=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_ort_profile(df, ax, title="test_title")
        fig.tight_layout()
        # fig.savefig("graph2.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    def _get_model_domain(self):
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Transpose", ["X"], ["Xt"], perm=[1, 0]),
                    make_node(
                        "CustomGemmFloat",
                        ["X", "Xt"],
                        ["final"],
                        domain="onnx_extented.ortops.tutorial.cpu",
                    ),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None, None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None, None])],
            ),
            opset_imports=[
                make_opsetid("", 18),
                make_opsetid("onnx_extented.ortops.tutorial.cpu", 1),
            ],
            ir_version=9,
        )
        check_model(model_def0)
        return model_def0

    @ignore_warnings(UserWarning)
    def test_plot_domain_agg(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess_options.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = InferenceSession(
            self._get_model_domain().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(16).astype(np.float32).reshape((-1, 4))))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_ort_profile(df, ax, title="test_title")
        fig.tight_layout()
        # fig.savefig("graph3.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    def _get_model2(self):
        model_def0 = make_model(
            make_graph(
                [
                    make_node("Add", ["X", "init1"], ["X1"]),
                    make_node("Abs", ["X"], ["X2"]),
                    make_node("Add", ["X", "init3"], ["inter"]),
                    make_node("Mul", ["X1", "inter"], ["Xm"]),
                    make_node("MatMul", ["X1", "Xm"], ["Xm2"]),
                    make_node("Sub", ["X2", "Xm2"], ["final"]),
                ],
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None, None])],
                [make_tensor_value_info("final", TensorProto.FLOAT, [None, None])],
                [
                    from_array(np.array([1], dtype=np.float32), name="init1"),
                    from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[make_opsetid("", 18)],
            ir_version=9,
        )
        check_model(model_def0)
        return model_def0

    @ignore_warnings(UserWarning)
    def test_plot_profile_timeline(self):
        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            self._get_model2().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.random.rand(2**10, 2**10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)

        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
        plot_ort_profile_timeline(df, ax, title="test_timeline", quantile=0.5)
        fig.tight_layout()
        fig.savefig("test_plot_profile_timeline.png")
        self.assertNotEmpty(fig)

        os.remove(prof)


if __name__ == "__main__":
    import logging

    for name in [
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
        "matplotlib",
        "matplotlib.pyplot",
    ]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    unittest.main(verbosity=2)
