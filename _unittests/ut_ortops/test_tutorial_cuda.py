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
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except ImportError:
    (
        SessionOptions,
        InferenceSession,
        get_available_providers,
        ort_version,
        OrtFail,
    ) = (
        None,
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

        if TensorProto.FLOAT8E4M3FN in tos or TensorProto.FLOAT8E5M2 in tos:
            gemm8 = True
            ir_version = 9
            opset = 19
        else:
            gemm8 = False
            ir_version = 8
            opset = 18

        casts = [make_node("Cast", [c], [c + "c"], to=to) for c, to in zip("AB", tos)]
        node_inputs = [c + "c" for c in "AB"]
        node_outputs = ["Yc"]
        if gemm8:
            node_inputs += ["scaleA", "scaleB"]
            node_outputs += ["scaleY"]
        nodes = [
            *casts,
            make_node(
                op_name,
                node_inputs,
                node_outputs,
                domain="onnx_extented.ortops.tutorial.cuda",
                transA=1,
                transB=0,
                alpha=kwargs.get("alpha", 1.0),
                **kwargs,
            ),
            make_node("Cast", ["Yc"], ["Y"], to=TensorProto.FLOAT),
        ]
        inputs = [
            make_tensor_value_info(c, TensorProto.FLOAT, [None, None]) for c in "AB"
        ]
        outputs = [make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])]
        if gemm8:
            inputs.extend(
                [
                    make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                ]
            )
            outputs.append(make_tensor_value_info("scaleY", TensorProto.FLOAT, [0]))
        graph = make_graph(nodes, "lr", inputs, outputs)
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
            (numpy.arange(256) / 256).astype(numpy.float32).reshape((-1, 16))
            for to in tos
        ]
        feeds = dict(zip("AB", inputs))
        if gemm8:
            feeds["scaleA"] = numpy.array([1], dtype=numpy.float32)
            feeds["scaleB"] = numpy.array([1], dtype=numpy.float32)
        try:
            got = sess.run(None, feeds)
        except OrtFail as e:
            dtypes = {k: v.dtype for k, v in feeds.items()}
            shapes = {k: v.shape for k, v in feeds.items()}
            raise AssertionError(
                f"Unable to run a model with dtypes={dtypes!r} "
                f"and shapes={shapes!r} "
                f"and model=\n{onnx_simple_text_plot(onnx_model)}."
            ) from e
        a, b = inputs[:2]
        if kwargs.get("rowMajor", 1):
            expected = a @ b.T
        else:
            expected = a.T @ b
        expected *= kwargs.get("alpha", 1.0)
        if gemm8:
            self.assertEqualArray(numpy.array([], numpy.float32), got[1])
            self.assertEqualArray(expected, got[0])
        else:
            self.assertEqualArray(expected, got[0])

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @unittest.skipIf(
        "CUDAExecutionProvider" not in get_available_providers(),
        reason="CUDA provider not available",
    )
    def test_custom_gemm_float32_default(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            fastAccumulationMode=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
        )

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @unittest.skipIf(
        "CUDAExecutionProvider" not in get_available_providers(),
        reason="CUDA provider not available",
    )
    def test_custom_gemm_float32_row_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            fastAccumulationMode=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=1,
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
        self.common_test_custom_gemm(
            "CustomGemmFloat8E4M3FN",
            [TensorProto.FLOAT8E4M3FN for i in range(2)],
            name="cgf8",
            fastAccumulationMode=1,
            rowMajor=0,
        )


if __name__ == "__main__":
    # TestOrtOpTutorialCuda().test_custom_gemm_float32()
    unittest.main(verbosity=2)
