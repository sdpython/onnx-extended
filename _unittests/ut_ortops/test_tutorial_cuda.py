import unittest
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
    tensor_dtype_to_np_dtype,
)

# from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnx_extended.ortops.tutorial.cuda import documentation

try:
    from onnxruntime import InferenceSession, SessionOptions, get_available_providers
except ImportError:
    SessionOptions, InferenceSession = None, None
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtOpTutorialCuda(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cuda import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)
        self.assertIn("cuda", r[0])

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 1)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    def common_test_custom_gemm(self, op_name, tos, **kwargs):
        from onnx_extended.ortops.tutorial.cuda import get_ort_ext_libs

        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        casts = [
            make_node("Cast", [c], [c + "c"], to=to) for c, to in zip("ABCDE", tos)
        ]
        nodes = [
            *casts,
            make_node(
                op_name,
                [c + "c" for c in "ABCDE"],
                ["Yc"],
                domain="onnx_extented.ortops.tutorial.cuda",
                transA=1,
                transB=0,
                alpha=kwargs.get("alpha", 1.0),
                beta=kwargs.get("beta", 0.0),
                **kwargs,
            ),
            make_node("Cast", ["Yc"], ["Y"], to=TensorProto.FLOAT),
        ]
        inputs = [
            make_tensor_value_info(c, TensorProto.FLOAT, [None, None]) for c in "ABCDE"
        ]
        graph = make_graph(nodes, "lr", inputs, [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("onnx_extented.ortops.tutorial.cuda", 1),
                make_opsetid("", 18),
            ],
            ir_version=8,
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inputs = [
            (numpy.arange(64) / 64)
            .astype(tensor_dtype_to_np_dtype(to))
            .reshape((-1, 8))
            for to in tos
        ]
        feeds = dict(zip("ABCDE", inputs))
        got = sess.run(None, feeds)[0]
        a, b, c = inputs[:3]
        expected = a.T @ b * kwargs.get("alpha", 1.0) + c * kwargs.get("beta", 0.0)
        self.assertEqualArray(expected, got)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @unittest.skipIf(
        "CUDAExecutionProvider" not in get_available_providers(),
        reason="CUDA provider not available",
    )
    def test_custom_gemm(self):
        tos = [TensorProto.FLOAT for i in range(5)]
        self.common_test_custom_gemm(
            "CustomGemmFloat", tos, name="cgf", fastAccumulationMode=1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
