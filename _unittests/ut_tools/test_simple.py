import os
import tempfile
import unittest
import numpy as np
from contextlib import redirect_stdout
from io import StringIO
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
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools import (
    enumerate_model_tensors,
    save_model,
    load_model,
    load_external,
)
from onnx_extended._command_lines import print_proto, display_intermediate_results
from onnx_extended.tools.onnx_manipulations import select_model_inputs_outputs


class TestSimple(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.INT64, [None, None])
        graph = make_graph(
            [
                make_node("Mul", ["X", "X"], ["X2"]),
                make_node("Add", ["X2", "Y"], ["z1"]),
                make_node("Mul", ["z1", "W"], ["z2"]),
                make_node("Cast", ["z2"], ["Z"], to=TensorProto.INT64),
            ],
            "add",
            [X],
            [Z],
            [
                from_array(np.arange(16).reshape((-1, 4)).astype(np.float32), name="Y"),
                from_array(
                    (np.arange(16).reshape((-1, 4)) + 100).astype(np.float32), name="W"
                ),
            ],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    def test_save_load_model(self):
        model = self._get_model()
        x = np.arange(16).reshape((-1, 4)).astype(np.float32)
        expected = CReferenceEvaluator(model).run(None, {"X": x})[0].tolist()
        with tempfile.TemporaryDirectory() as root:
            name = os.path.join(root, "model_full.onnx")
            save_model(model, name, external=False)
            self.assertExists(name)
            self.assertEqual(os.listdir(root), ["model_full.onnx"])
            model2 = load_model(name)
            self.assertEqual(model.SerializeToString(), model2.SerializeToString())
            self.assertEqual(
                expected, CReferenceEvaluator(model2).run(None, {"X": x})[0].tolist()
            )

            name = os.path.join(root, "model_ext.onnx")
            save_model(model, name, external=True, size_threshold=15)
            self.assertEqual(
                list(sorted(os.listdir(root))),
                ["model_ext.onnx", "model_ext.onnx.data", "model_full.onnx"],
            )
            model3 = load_model(name)
            self.assertNotEqual(model.SerializeToString(), model3.SerializeToString())
            model4 = load_model(name, external=True)
            self.assertEqual(
                expected, CReferenceEvaluator(model4).run(None, {"X": x})[0].tolist()
            )

    def test_print(self):
        model = self._get_model()
        with tempfile.TemporaryDirectory() as root:
            name = os.path.join(root, "model_ext.onnx")
            save_model(model, name, external=True, size_threshold=15)
            self.assertEqual(
                list(sorted(os.listdir(root))),
                ["model_ext.onnx", "model_ext.onnx.data"],
            )
            st = StringIO()
            with redirect_stdout(st):
                print_proto(name, external=False)
            text = st.getvalue()
            self.assertIn('value: "model_ext.onnx.data"', text)
            self.assertIn('key: "offset"', text)

    def test_display_intermediate_results(self):
        model = self._get_model()
        with tempfile.TemporaryDirectory() as root:
            name = os.path.join(root, "model_ext.onnx")
            save_model(model, name, external=True, size_threshold=15)
            self.assertEqual(
                list(sorted(os.listdir(root))),
                ["model_ext.onnx", "model_ext.onnx.data"],
            )
            st = StringIO()
            with redirect_stdout(st):
                display_intermediate_results(name, external=False)
            text = st.getvalue()
            self.assertIn("unk__0xunk_", text)

    def test_select_model_inputs_outputs(self):
        model = self._get_model()
        with tempfile.TemporaryDirectory() as root:
            name = os.path.join(root, "model_ext.onnx")
            save_model(model, name, external=True, size_threshold=15)
            self.assertEqual(
                list(sorted(os.listdir(root))),
                ["model_ext.onnx", "model_ext.onnx.data"],
            )

            # X
            name2 = os.path.join(root, "sub_model_ext.onnx")
            model2 = load_model(name, external=False)
            new_model = select_model_inputs_outputs(model2, outputs=["X2"])
            save_model(new_model, name2)

            x = np.arange(16).reshape((-1, 4)).astype(np.float32)
            y = np.arange(16).reshape((-1, 4)).astype(np.float32)

            sess = CReferenceEvaluator(new_model)
            got = sess.run(None, {"X": x})[0]
            self.assertEqual((x**2).tolist(), got.tolist())

            sess = CReferenceEvaluator(name2)
            got = sess.run(None, {"X": x})[0]
            self.assertEqual((x**2).tolist(), got.tolist())

            # z1
            name3 = os.path.join(root, "sub_model_ext_z1.onnx")
            model2 = load_model(name, external=False)
            new_model = select_model_inputs_outputs(model2, outputs=["z1"])
            save_model(new_model, name3)
            self.assertEqual(
                [
                    "model_ext.onnx",
                    "model_ext.onnx.data",
                    "sub_model_ext.onnx",
                    "sub_model_ext_z1.onnx",
                ],
                list(sorted(os.listdir(root))),
            )

            x = np.arange(16).reshape((-1, 4)).astype(np.float32)

            sess = CReferenceEvaluator(name3)
            got = sess.run(None, {"X": x})[0]
            self.assertEqual((x**2 + y).tolist(), got.tolist())

            tensors = list(enumerate_model_tensors(new_model))
            self.assertEqual(len(tensors), 1)
            self.assertIsInstance(tensors[0], tuple)
            self.assertEqual(len(tensors[0]), 2)
            self.assertTrue(tensors[0][-1])
            self.assertIsInstance(tensors[0][0], TensorProto)
            load_external(new_model, root)
            sess = CReferenceEvaluator(new_model)
            got = sess.run(None, {"X": x})[0]
            self.assertEqual((x**2 + y).tolist(), got.tolist())


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
