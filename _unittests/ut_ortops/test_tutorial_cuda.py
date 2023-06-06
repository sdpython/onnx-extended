import unittest
import numpy
from packaging.version import Version
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model

try:
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
except ImportError:
    onnx_simple_text_plot = str
from onnx_extended.ortops.tutorial.cuda import documentation

try:
    from onnxruntime import (
        InferenceSession,
        SessionOptions,
        get_available_providers,
        __version__ as ort_version,
    )
except ImportError:
    SessionOptions, InferenceSession, get_available_providers, ort_version = (
        None,
        None,
        None,
        None,
    )
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

        if TensorProto.FLOAT8E4M3FN in tos or TensorProto.FLOAT8E5M2:
            ir_version = 9
            opset = 19
        else:
            ir_version = 8
            opset = 18

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
                make_opsetid("", opset),
            ],
            ir_version=ir_version,
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        try:
            sess = InferenceSession(
                onnx_model.SerializeToString(),
                opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to create InferenceSession with "
                f"onx={onnx_simple_text_plot(onnx_model)}"
            ) from e
        inputs = [
            (numpy.arange(64) / 64).astype(numpy.float32).reshape((-1, 8)) for to in tos
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
    def test_custom_gemm_float32(self):
        tos = [TensorProto.FLOAT for i in range(5)]
        self.common_test_custom_gemm(
            "CustomGemmFloat", tos, name="cgf", fastAccumulationMode=1
        )

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @unittest.skipIf(
        "CUDAExecutionProvider" not in get_available_providers(),
        reason="CUDA provider not available",
    )
    @unittest.skipIf(
        Version(ort_version) < Version("1.16"), reason="float8 types not released"
    )
    def test_custom_gemm_float8(self):
        tos = [TensorProto.FLOAT8E4M3FN for i in range(5)]
        tos[3] = TensorProto.FLOAT16
        self.common_test_custom_gemm(
            "CustomGemmFloat8E4M3FN", tos, name="cgf8", fastAccumulationMode=1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
