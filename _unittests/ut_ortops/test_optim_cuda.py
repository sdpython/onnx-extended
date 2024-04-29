import unittest
import numpy as np
from onnx import TensorProto
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_extended.ortops.tutorial.cpu import documentation
from onnx_extended.reference import CReferenceEvaluator

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda


class TestOrtOpOptimCuda(ExtTestCase):
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 4)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    def _scatternd_of_shape_cuda(self, reduction, line, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterND",
                        inputs=["data", "indices", "updates"],
                        outputs=["y"],
                        reduction=reduction,
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("data", itype, [None, None, None]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info("updates", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("y", itype, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        data = np.zeros((2, 2, 3), dtype=dtype)

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterNDOfShape",
                        inputs=["shape", "indices", "updates"],
                        outputs=["y"],
                        reduction=reduction,
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info("updates", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("y", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        indices = np.array([[line], [1 - line], [line]], dtype=np.int64)
        if itype == TensorProto.FLOAT:
            updates = (2 ** np.arange(18).reshape((3, 2, 3))).astype(dtype)
        else:
            updates = np.arange(18).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(data=data, indices=indices, updates=updates)
        feeds2 = dict(
            shape=np.array([2, 2, 3], dtype=np.int64), indices=indices, updates=updates
        )
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        # opts.log_severity_level = 0
        # opts.log_verbosity_level = 0
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds2)[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_scatternd_of_shape_standalone_cuda(self):
        self._scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT)
        self._scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT16)
        self._scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT)
        self._scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT16)

    def _addaddmulmul_cuda(self, itype, op_type, broad=False):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["X", "Y"], ["xy"]),
                    oh.make_node(op_type, ["xy", "Z"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type}{op_type}",
                        ["X", "Y", "Z"],
                        ["final"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        shapex = (1, 2, 3) if broad else (3, 2, 3)
        shapey = (3, 2, 3)
        shapez = (1, 2, 3) if broad else (3, 2, 3)
        x = (np.arange(np.prod(shapex)) + 1).reshape(shapex).astype(dtype)
        y = (np.arange(np.prod(shapey)) + 1).reshape(shapey).astype(dtype)
        z = (np.arange(np.prod(shapez)) + 1).reshape(shapez).astype(dtype)

        feeds1 = dict(X=x, Y=y, Z=z)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmul_cuda(self):
        self._addaddmulmul_cuda(TensorProto.FLOAT, "Mul")
        self._addaddmulmul_cuda(TensorProto.FLOAT16, "Mul")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmul_cuda_broadcast(self):
        self._addaddmulmul_cuda(TensorProto.FLOAT, "Mul", True)
        self._addaddmulmul_cuda(TensorProto.FLOAT16, "Mul", True)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_addadd_cuda(self):
        self._addaddmulmul_cuda(TensorProto.FLOAT, "Add")
        self._addaddmulmul_cuda(TensorProto.FLOAT16, "Add")

    def _scatternd_of_shape_optimize_cuda(self, optimize, dim3, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        indices_shape = ["i", "j", 1] if dim3 else ["j", 1]
        updates_shape = ["i", "j", "b"] if dim3 else ["j", "b"]

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterNDOfShape",
                        inputs=["shape", "indices", "updates"],
                        outputs=["y"],
                        reduction="add",
                        strategy="optimize" if optimize else "none",
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [2]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, indices_shape
                    ),
                    oh.make_tensor_value_info("updates", itype, updates_shape),
                ],
                [oh.make_tensor_value_info("y", itype, [None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        if dim3:
            shape = (128, 1024)
            indices = np.zeros((2, 64, 1)).astype(np.int64)
            indices[:, ::2, 0] = 87
            indices[:, ::3, 0] = 85
            updates = np.ones((2, 64, 1024)).astype(np.float32)
        else:
            shape = (128, 1024)
            indices = np.zeros((128, 1)).astype(np.int64)
            indices[::2, 0] = 87
            indices[::3, 0] = 85
            updates = np.ones((128, 1024)).astype(np.float32)
        if itype != 1:
            updates = updates.astype(np.float16)
        feeds = dict(
            shape=np.array(shape, dtype=np.int64), indices=indices, updates=updates
        )

        ref = CReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        if __name__ == "disabled__main__":
            opts.log_severity_level = 0
            opts.log_verbosity_level = 0
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        if __name__ == "disabled__main__":
            print(
                f"running itype={itype}, optimize={optimize}, dim3={dim3}, "
                f"shape={shape}, indices.shape={indices.shape}, "
                f"updates.shape={updates.shape}"
            )
            ro = onnxruntime.RunOptions()
            ro.log_severity_level = 0
            ro.log_verbosity_level = 0
        else:
            ro = None
        got = sess.run(None, feeds, ro)[0]
        self.assertEqual(expected.tolist(), got.tolist())
        if __name__ == "disabled__main__":
            print("done.")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_scatternd_of_shape_optimize_cuda(self):
        with self.subTest(optimize=True, dim3=True):
            self._scatternd_of_shape_optimize_cuda(True, True, TensorProto.FLOAT)
        self._scatternd_of_shape_optimize_cuda(False, False, TensorProto.FLOAT)
        self._scatternd_of_shape_optimize_cuda(False, True, TensorProto.FLOAT)
        with self.subTest(optimize=True, dim3=False):
            self._scatternd_of_shape_optimize_cuda(True, False, TensorProto.FLOAT)
        with self.subTest(optimize=True, dim3=True, itype=TensorProto.FLOAT16):
            self._scatternd_of_shape_optimize_cuda(True, True, TensorProto.FLOAT16)

    def _addaddaddmulmulmul_cuda(self, itype, op_type, broad=False):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["X", "Y"], ["xy"]),
                    oh.make_node(op_type, ["xy", "Z"], ["xyz"]),
                    oh.make_node(op_type, ["xyz", "W"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                    oh.make_tensor_value_info("W", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type}{op_type}{op_type}",
                        ["X", "Y", "Z", "W"],
                        ["final"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                    oh.make_tensor_value_info("W", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        shapex = (1, 2, 3) if broad else (3, 2, 3)
        shapey = (3, 2, 3)
        shapez = (1, 2, 3) if broad else (3, 2, 3)
        shapew = (3, 2, 3)
        x = ((np.arange(np.prod(shapex)) + 1) / 10).reshape(shapex).astype(dtype)
        y = ((np.arange(np.prod(shapey)) + 1) / 10).reshape(shapey).astype(dtype)
        z = (np.arange(np.prod(shapez)) + 1).reshape(shapez).astype(dtype)
        w = (np.arange(np.prod(shapew)) + 1).reshape(shapew).astype(dtype)

        feeds1 = dict(X=x, Y=y, Z=z, W=w)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(
            expected, got, rtol=1e-3 if itype == TensorProto.FLOAT16 else 1e-5
        )

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmulmul_cuda(self):
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT, "Mul")
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT16, "Mul")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmulmul_cuda_broadcast(self):
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT, "Mul", True)
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT16, "Mul", True)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_addaddadd_cuda(self):
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT, "Add")
        self._addaddaddmulmulmul_cuda(TensorProto.FLOAT16, "Add")

    def _addmul_cuda(self, itype, op_type1, op_type2, broad=False):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type1, ["X", "Y"], ["xy"]),
                    oh.make_node(op_type2, ["xy", "Z"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type1}{op_type2}",
                        ["X", "Y", "Z"],
                        ["final"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        shapex = (1, 2, 3) if broad else (3, 2, 3)
        shapey = (3, 2, 3)
        shapez = (1, 2, 3) if broad else (3, 2, 3)
        x = (np.arange(np.prod(shapex)) + 1).reshape(shapex).astype(dtype)
        y = (np.arange(np.prod(shapey)) + 1).reshape(shapey).astype(dtype)
        z = (np.arange(np.prod(shapez)) + 1).reshape(shapez).astype(dtype)

        feeds1 = dict(X=x, Y=y, Z=z)
        ref = CReferenceEvaluator(model1, verbose=0)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(expected, got)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_addmul_cuda(self):
        self._addmul_cuda(TensorProto.FLOAT, "Add", "Mul")
        self._addmul_cuda(TensorProto.FLOAT16, "Add", "Mul")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_addmul_cuda_broadcast(self):
        self._addmul_cuda(TensorProto.FLOAT, "Add", "Mul", True)
        self._addmul_cuda(TensorProto.FLOAT16, "Add", "Mul", True)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_muladd_cuda(self):
        self._addmul_cuda(TensorProto.FLOAT, "Mul", "Add")
        self._addmul_cuda(TensorProto.FLOAT16, "Mul", "Add")

    def _rotary_cuda(self, itype, side):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Rotary",
                        ["X", "splits"],
                        ["Y"],
                        domain="onnx_extended.ortops.optim.cuda",
                        side=side,
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None, None]),
                    oh.make_tensor_value_info("splits", TensorProto.INT64, [2]),
                ],
                [oh.make_tensor_value_info("Y", itype, [None, None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(18 * 4) + 1).reshape((3, 2, 3, 4)).astype(dtype)
        splits = np.array([x.shape[-1] // 2, x.shape[-1] // 2], dtype=np.int64)

        expected = x.copy()
        half = x.shape[-1] // 2
        if side == "right":
            expected[:, :, :, :half] = x[:, :, :, half:]
            expected[:, :, :, half:] = -x[:, :, :, :half]
        else:
            expected[:, :, :, :half] = -x[:, :, :, half:]
            expected[:, :, :, half:] = x[:, :, :, :half]

        feeds = dict(X=x, splits=splits)
        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_rotary_cuda(self):
        self._rotary_cuda(TensorProto.FLOAT, "left")
        self._rotary_cuda(TensorProto.FLOAT16, "left")
        self._rotary_cuda(TensorProto.FLOAT, "right")
        self._rotary_cuda(TensorProto.FLOAT16, "right")

    def _mul_sigmoid_cuda(self, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["X"], ["sx"]),
                    oh.make_node("Mul", ["X", "sx"], ["Y"]),
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "MulSigmoid",
                        ["X"],
                        ["Y"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(18) + 1).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(
            expected, got, atol=1e-5 if itype == TensorProto.FLOAT else 1e-2
        )

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mul_sigmoid_cuda(self):
        self._mul_sigmoid_cuda(TensorProto.FLOAT)
        self._mul_sigmoid_cuda(TensorProto.FLOAT16)

    def _replace_zero_cuda(self, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Equal", ["X", "zero"], ["cond"]),
                    oh.make_node("Where", ["cond", "cst", "X"], ["Y"]),
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
                [
                    onh.from_array(np.array([0], dtype=dtype), name="zero"),
                    onh.from_array(np.array([1.67], dtype=dtype), name="cst"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ReplaceZero",
                        ["X"],
                        ["Y"],
                        by=1.67,
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(18) - 4).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_replace_zero_cuda(self):
        self._replace_zero_cuda(TensorProto.FLOAT)
        self._replace_zero_cuda(TensorProto.FLOAT16)

    def _tri_matrix_cuda(self, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "TriMatrix",
                        ["shape", "csts"],
                        ["final"],
                        domain="onnx_extended.ortops.optim.cuda",
                    ),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [2]),
                    oh.make_tensor_value_info("csts", itype, [3]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        shape = np.array([8, 8], dtype=np.int64)
        csts = np.array([2, 3, 4], dtype=dtype)
        expected = np.empty((8, 8), dtype=dtype)
        i1 = np.arange(8).reshape((-1, 1))
        i2 = np.arange(8).reshape((1, -1))
        expected[i1 < i2] = 4
        expected[i1 == i2] = 3
        expected[i1 > i2] = 2
        feeds = dict(shape=shape, csts=csts)

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_tri_matrix_cuda(self):
        self._tri_matrix_cuda(TensorProto.FLOAT)
        self._tri_matrix_cuda(TensorProto.FLOAT16)

    def _negxplus1_cuda(self, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sub", ["one", "X"], ["Y"]),
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
                [onh.from_array(np.array([1], dtype=dtype), name="one")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "NegXplus1",
                        ["X"],
                        ["Y"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None, None])],
                [oh.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(18) - 4).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_negxplus1_cuda(self):
        self._negxplus1_cuda(TensorProto.FLOAT)
        self._negxplus1_cuda(TensorProto.FLOAT16)

    def _transpose_cast_cuda(self, itype):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        itype2 = (
            TensorProto.FLOAT if itype == TensorProto.FLOAT16 else TensorProto.FLOAT16
        )
        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Transpose", ["X"], ["t"], perm=[1, 0]),
                    oh.make_node("Cast", ["t"], ["Y"], to=itype2),
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None])],
                [oh.make_tensor_value_info("Y", itype2, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        (
                            "Transpose2DCastFP16"
                            if itype2 == TensorProto.FLOAT16
                            else "Transpose2DCastFP32"
                        ),
                        ["X"],
                        ["Y"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [oh.make_tensor_value_info("X", itype, [None, None])],
                [oh.make_tensor_value_info("Y", itype2, [None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(32 * 32 * 3) + 1).reshape((32, 32 * 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_transpose_cast_cuda(self):
        self._transpose_cast_cuda(TensorProto.FLOAT)
        self._transpose_cast_cuda(TensorProto.FLOAT16)

    def _addmul_shared_input_cuda(
        self, itype, op_type, shapea=(3, 2, 3), shapeb=(3, 2, 3), shapec=(3, 2, 3)
    ):
        import onnxruntime
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        model1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(op_type, ["X", "Y"], ["XY"]),
                    oh.make_node(op_type, ["X", "Z"], ["XZ"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [
                    oh.make_tensor_value_info("XY", itype, [None, None, None]),
                    oh.make_tensor_value_info("XZ", itype, [None, None, None]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        f"{op_type}SharedInput",
                        ["X", "Y", "Z"],
                        ["XY", "XZ"],
                        domain="onnx_extended.ortops.optim.cuda",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [
                    oh.make_tensor_value_info("XY", itype, [None, None, None]),
                    oh.make_tensor_value_info("XZ", itype, [None, None, None]),
                ],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(np.prod(shapea)) + 1).reshape((shapea)).astype(dtype)
        y = (np.arange(np.prod(shapeb)) + 2).reshape((shapeb)).astype(dtype)
        z = (np.arange(np.prod(shapec)) + 3).reshape((shapec)).astype(dtype)

        feeds1 = dict(X=x, Y=y, Z=z)
        ref = CReferenceEvaluator(model1, verbose=0)
        expected = ref.run(None, feeds1)

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(get_ort_ext_libs()[0])
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds1)
        for i in range(2):
            self.assertEqualArray(expected[i], got[i])

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_add_shared_input_cuda(self):
        self._addmul_shared_input_cuda(TensorProto.FLOAT, "Add")
        self._addmul_shared_input_cuda(TensorProto.FLOAT16, "Add")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mul_shared_input_cuda(self):
        self._addmul_shared_input_cuda(TensorProto.FLOAT, "Mul")
        self._addmul_shared_input_cuda(TensorProto.FLOAT16, "Mul")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_add_shared_input_cuda_broadcast1(self):
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT,
            "Add",
            shapea=(3, 2, 3),
            shapeb=(1, 2, 3),
            shapec=(1, 2, 3),
        )
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT16,
            "Add",
            shapea=(3, 2, 3),
            shapeb=(1, 2, 3),
            shapec=(1, 2, 3),
        )

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_add_shared_input_cuda_broadcast2(self):
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT,
            "Add",
            shapea=(1, 2, 3),
            shapeb=(3, 2, 3),
            shapec=(3, 2, 3),
        )
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT16,
            "Add",
            shapea=(1, 2, 3),
            shapeb=(3, 2, 3),
            shapec=(3, 2, 3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
