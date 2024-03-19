import unittest
import numpy as np
from onnx import TensorProto
import onnx.helper as oh
from onnx_extended.ortops.tutorial.cpu import documentation
from onnx_extended.reference import CReferenceEvaluator

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda


class TestOrtOpOptimCuda(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 4)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    def _scatternd_of_shape_cuda(self, reduction, line):
        import onnxruntime

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
                    oh.make_tensor_value_info(
                        "data", TensorProto.FLOAT, [None, None, None]
                    ),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info(
                        "updates", TensorProto.FLOAT, [None, None, None]
                    ),
                ],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        data = np.zeros((2, 2, 3), dtype=np.float32)

        model2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterNDOfShape",
                        inputs=["shape", "indices", "updates"],
                        outputs=["y"],
                        reduction=reduction,
                        domain="com.microsoft",
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    oh.make_tensor_value_info(
                        "indices", TensorProto.INT64, [None, None]
                    ),
                    oh.make_tensor_value_info(
                        "updates", TensorProto.FLOAT, [None, None, None]
                    ),
                ],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        indices = np.array([[line], [1 - line], [line]], dtype=np.int64)
        updates = (2 ** np.arange(18).astype(np.float32).reshape((3, 2, 3))).astype(
            np.float32
        )

        feeds1 = dict(data=data, indices=indices, updates=updates)
        feeds2 = dict(
            shape=np.array([2, 2, 3], dtype=np.int64), indices=indices, updates=updates
        )
        ref = CReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        # opts.log_severity_level = 0
        # opts.log_verbosity_level = 0
        sess = onnxruntime.InferenceSession(
            model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        got = sess.run(None, feeds2)[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_scatternd_of_shape_standalone_cuda(self):
        self._scatternd_of_shape_cuda("add", 0)
        self._scatternd_of_shape_cuda("add", 1)


if __name__ == "__main__":
    # TestOrtOpTutorialCpu().test_dynamic_quantize_linear()
    unittest.main(verbosity=2)
