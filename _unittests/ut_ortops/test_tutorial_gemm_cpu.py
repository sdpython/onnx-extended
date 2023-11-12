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
    make_tensor,
)
from onnx.checker import check_model

try:
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
except ImportError:
    onnx_simple_text_plot = str
try:
    from onnxruntime import InferenceSession, __version__ as ort_version
except ImportError:
    InferenceSession = None
    ort_version = "0.0"
if InferenceSession is not None:
    from onnxruntime import SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail


from onnx_extended.ortops.tutorial.cpu import documentation
from onnx_extended.helper import make_dynamic_quantize_linear_function_proto
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import get_stdcpp


class TestOrtOpTutorialCpu(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)
        self.assertIn("cpu", r[0])

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 3)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    def common_test_custom_gemm(
        self, op_name, tos, return_sess=False, square=True, **kwargs
    ):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        add_time = op_name.endswith("Time")
        if TensorProto.FLOAT8E4M3FN in tos or TensorProto.FLOAT8E5M2 in tos:
            gemm8 = True
            ir_version = 9
            opset = 19
        else:
            gemm8 = False
            ir_version = 8
            opset = 18

        bias = kwargs.get("beta", 0) != 0
        input_names = "ABC" if bias else "AB"
        self.assertEqual(len(input_names), len(tos))
        casts = [
            make_node("Cast", [c], [c + "c"], to=to) for c, to in zip(input_names, tos)
        ]
        node_inputs = [c + "c" for c in input_names]
        node_outputs = ["Yc", "time"] if add_time else ["Yc"]
        if gemm8:
            if len(tos) == 2:
                node_inputs.append("")
            node_inputs.extend(["scaleA", "scaleB", "scaleY"])
        nodes = [
            *casts,
            make_node(
                op_name,
                node_inputs,
                node_outputs,
                domain="onnx_extented.ortops.tutorial.cpu",
                **kwargs,
            ),
            make_node("Cast", ["Yc"], ["Y"], to=TensorProto.FLOAT),
        ]
        inputs = [
            make_tensor_value_info(c, TensorProto.FLOAT, [None, None])
            for c in input_names
        ]
        outputs = [make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])]
        if add_time:
            outputs.append(make_tensor_value_info("time", TensorProto.DOUBLE, [None]))
        if gemm8:
            inputs.extend(
                [
                    make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                    make_tensor_value_info("scaleY", TensorProto.FLOAT, [1]),
                ]
            )
        graph = make_graph(nodes, "lr", inputs, outputs)
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("onnx_extented.ortops.tutorial.cpu", 1),
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
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to create InferenceSession with "
                f"onx={onnx_simple_text_plot(onnx_model)}"
            ) from e
        if return_sess:
            return onnx_model, sess

        if square:
            inputs = [
                (numpy.arange(16) / 16).astype(numpy.float32).reshape((-1, 4))
                for to in tos
            ]
            inputs[0][0, 1] = -0.5
            inputs[1][0, 1] = -0.145
        else:
            inputs = [
                (numpy.arange(10) / 10).astype(numpy.float32).reshape((2, -1)),
                (numpy.arange(15) / 15).astype(numpy.float32).reshape((5, -1)),
            ]
            inputs[0][0, 1] = -0.5
            inputs[1][0, 1] = -0.145
            if len(tos) == 3:
                inputs.append(
                    (numpy.arange(6) / 6).astype(numpy.float32).reshape((2, 3))
                )
            if kwargs.get("transA", 0):
                inputs[0] = inputs[0].T
            if kwargs.get("transB", 0):
                inputs[1] = inputs[1].T

        a, b = inputs[:2]
        expected = (a.T if kwargs.get("transA", 0) else a) @ (
            b.T if kwargs.get("transB", 0) else b
        )
        expected *= kwargs.get("alpha", 1.0)
        if bias:
            expected += inputs[2] * kwargs.get("beta", 0)

        feeds = dict(zip(input_names, inputs))
        if gemm8:
            feeds["scaleA"] = numpy.array([1], dtype=numpy.float32)
            feeds["scaleB"] = numpy.array([1], dtype=numpy.float32)
            feeds["scaleY"] = numpy.array([1], dtype=numpy.float32)
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

        if tos[0] == TensorProto.FLOAT16:
            atol = 1e-2
        else:
            atol = 0.08 if gemm8 else 1e-6
        try:
            self.assertEqualArray(expected, got[0], atol=atol)
        except Exception as e:

            def check(f):
                try:
                    return f()[:2, :2]
                except Exception as e:
                    return str(e)

            raise AssertionError(
                f"ERROR len(inputs)={len(inputs)}"
                f"\na@b=\n{check(lambda:a@b)}"
                f"\na.T@b=\n{check(lambda:a.T@b)}"
                f"\na@b.T=\n{check(lambda:a@b.T)}"
                f"\na.T@b.T=\n{check(lambda:a.T@b.T)}"
                f"\n----\nb@a=\n{check(lambda:b@a)}"
                f"\nb.T@a=\n{check(lambda:b.T@a)}"
                f"\nb@a.T=\n{check(lambda:b@a.T)}"
                f"\nb.T@a.T=\n{check(lambda:b.T@a.T)}"
                f"\n----\nexpected=\n{expected[:2,:2]}"
                f"\n----\ngot=\n{got[0][:2,:2]}"
            ) from e

    def test_custom_gemm_base0_cnot_square(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=1,
            square=False,
        )

    def test_custom_gemm_base0_no_trans(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
        )

    def test_custom_gemm_base0_no_trans_bias(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
        )

    def test_custom_gemm_base2_no_trans_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
        )

    def test_custom_gemm_base2_no_trans_col_major_bias(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
            beta=1.0,
        )

    def test_custom_gemm_base0_with_transa(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transA=1,
        )

    def test_custom_gemm_base2_with_transa_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transA=1,
            rowMajor=0,
        )

    def test_custom_gemm_base2_with_transb_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transB=1,
            rowMajor=0,
        )

    def test_custom_gemm_base2_with_transab_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transA=1,
            transB=1,
            rowMajor=0,
        )

    def test_custom_gemm_base0_with_transb(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transB=1,
        )

    def test_custom_gemm_base0_with_transab(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transA=1,
            transB=1,
        )

    def test_custom_gemm_basic_float32_default(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
        )

    def test_custom_gemm_float32_relu(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            activation="RELU",
        )

    def test_custom_gemm_float32_gelu(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            activation="GELU",
        )

    def test_custom_gemm_float32_col_major_relu(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            activation="RELU",
            rowMajor=0,
        )

    def test_custom_gemm_float32_col_major_gelu(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            activation="GELU",
            rowMajor=0,
        )

    def test_custom_gemm_float32_not_square(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            square=False,
        )

    def test_custom_gemm_float32_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
        )

    def test_custom_gemm_base0_col_major_not_square(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
            square=False,
        )

    def test_custom_gemm_base0_col_major_not_square_with_transa(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
            square=False,
        )

    def test_custom_gemm_base0_col_major_not_square_with_transb(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transB=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
            square=False,
        )

    def test_custom_gemm_base0_col_major_not_square_with_transab(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            transB=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            rowMajor=0,
            square=False,
        )

    def test_custom_gemm_basic_float32_bias(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
        )

    def test_custom_gemm_float32_bias_01(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            transB=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
        )

    def test_custom_gemm_float32_bias_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
            rowMajor=0,
        )

    def test_custom_gemm_float32_not_square_bias(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
            square=False,
        )

    def test_custom_gemm_float32_not_square_bias_col_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(3)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            beta=1.0,
            square=False,
            rowMajor=0,
        )

    @unittest.skipIf(get_stdcpp() < 23, "Gemm for float16 not implemented for CPU.")
    def test_custom_gemm_float16_default(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat16",
            [TensorProto.FLOAT16 for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F",
        )

    def test_custom_gemm_float32_row_major(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
            transA=1,
            rowMajor=1,
        )

    @unittest.skipIf(
        Version(ort_version) < Version("1.17.0"), reason="type inference failed"
    )
    def test_custom_gemm_float8(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat8E4M3FN",
            [TensorProto.FLOAT8E4M3FN for i in range(2)],
            name="cgf8",
            transB=1,
            rowMajor=0,
        )

    @unittest.skipIf(
        Version(ort_version) < Version("1.17.0"), reason="type inference failed"
    )
    def test_custom_gemm_float8_not_square(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat8E4M3FN",
            [TensorProto.FLOAT8E4M3FN for i in range(2)],
            name="cgf8",
            transB=1,
            rowMajor=0,
            square=False,
        )

    def common_test_custom_gemm_cast(self, op_name, tos, return_sess=False, **kwargs):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        gemm8 = False
        ir_version = 8
        opset = 18

        bias = kwargs.get("beta", 0) != 0
        input_names = "ABC" if bias else "AB"
        self.assertEqual(len(input_names), len(tos))
        casts = [
            make_node("Cast", [c], [c + "c"], to=to) for c, to in zip(input_names, tos)
        ]
        node_inputs = [c + "c" for c in input_names]
        node_outputs = ["Yc"]
        if gemm8:
            if len(tos) == 2:
                node_inputs.append("")
            node_inputs.extend(["scaleA", "scaleB", "scaleY"])
        nodes = [
            *casts,
            make_node(
                "Constant",
                [],
                ["new_shape"],
                value=make_tensor("new_shape", TensorProto.INT64, [1], [-1]),
            ),
            make_node(
                op_name,
                node_inputs,
                node_outputs,
                domain="onnx_extented.ortops.tutorial.cpu",
                **kwargs,
            ),
            make_node("Reshape", ["Yc", "new_shape"], ["Yr"]),
            make_node("Cast", ["Yr"], ["Y"], to=TensorProto.INT64),
        ]
        inputs = [
            make_tensor_value_info(c, TensorProto.FLOAT, [None, None])
            for c in input_names
        ]
        outputs = [make_tensor_value_info("Y", TensorProto.INT64, [None])]
        if gemm8:
            inputs.extend(
                [
                    make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                    make_tensor_value_info("scaleY", TensorProto.FLOAT, [1]),
                ]
            )
        graph = make_graph(nodes, "lr", inputs, outputs)
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("onnx_extented.ortops.tutorial.cpu", 1),
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
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to create InferenceSession with "
                f"onx={onnx_simple_text_plot(onnx_model)}"
            ) from e
        if return_sess:
            return onnx_model, sess

        inputs = [
            (numpy.arange(256) / 256).astype(numpy.float32).reshape((32, -1)),
            (numpy.arange(512) / 512).astype(numpy.float32).reshape((32, -1)),
        ]
        if len(tos) == 3:
            inputs.append(
                (numpy.arange(128) / 128).astype(numpy.float32).reshape((8, 16))
            )

        a, b = inputs[:2]
        expected = (a.T if kwargs.get("transA", 0) else a) @ (
            b.T if kwargs.get("transB", 0) else b
        )
        expected *= kwargs.get("alpha", 1.0)
        if bias:
            expected += inputs[2] * kwargs.get("beta", 0)
        expected = expected.reshape((-1,)).astype(numpy.int64)

        feeds = dict(zip(input_names, inputs))
        if gemm8:
            feeds["scaleA"] = numpy.array([1], dtype=numpy.float32)
            feeds["scaleB"] = numpy.array([1], dtype=numpy.float32)
            feeds["scaleY"] = numpy.array([1], dtype=numpy.float32)
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

        try:
            self.assertEqualArray(expected, got[0])
        except Exception as e:

            def check(f):
                try:
                    return f()[:2, :2]
                except Exception as e:
                    return str(e)

            raise AssertionError(
                f"ERROR len(inputs)={len(inputs)}"
                f"\na@b=\n{check(lambda:a@b)}"
                f"\na.T@b=\n{check(lambda:a.T@b)}"
                f"\na@b.T=\n{check(lambda:a@b.T)}"
                f"\na.T@b.T=\n{check(lambda:a.T@b.T)}"
                f"\n----\nb@a=\n{check(lambda:b@a)}"
                f"\nb.T@a=\n{check(lambda:b.T@a)}"
                f"\nb@a.T=\n{check(lambda:b@a.T)}"
                f"\nb.T@a.T=\n{check(lambda:b.T@a.T)}"
                f"\n----\nexpected=\n{expected[:2,:2]}"
                f"\n----\ngot=\n{got[0][:2,:2]}"
            ) from e

    def test_custom_gemm_float32_default_cast(self):
        self.common_test_custom_gemm_cast(
            "CustomGemmFloat",
            [TensorProto.FLOAT for i in range(2)],
            name="cgf",
            transA=1,
            computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
        )

    def _get_model_dql(self, use_local):
        nodes = [
            make_node("Identity", ["X"], ["x"]),
            make_node(
                "DynamicQuantizeLinear",
                ["x"],
                ["y", "ScaleScaled", "Zeropoint"],
                to=TensorProto.FLOAT8E4M3FN,
                domain="local",
            ),
            make_node(
                "CustomGemmFloat8E4M3FN",
                # "GemmFloat8",
                ["y", "y", "", "ScaleScaled", "ScaleScaled", ""],
                ["Yf"],
                domain="onnx_extented.ortops.tutorial.cpu",
                # domain="com.microsoft",
                dtype=1,
                transB=1,
                rowMajor=1,
                computeType="CUBLAS_COMPUTE_32F",
            ),
            make_node("Cast", ["Yf"], ["Y"], to=TensorProto.FLOAT),
        ]

        if use_local:
            functions = [make_dynamic_quantize_linear_function_proto("local", 19)]
        else:
            dql = make_dynamic_quantize_linear_function_proto(
                "local", 19, to=TensorProto.FLOAT8E4M3FN
            )
            functions = []
            nodes = nodes[:1] + list(dql.node) + nodes[2:]

        onnx_model = make_model(
            make_graph(
                nodes,
                "test",
                [make_tensor_value_info("X", TensorProto.FLOAT, [None, None])],
                [make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])],
            ),
            opset_imports=[
                make_opsetid("", 19),
                make_opsetid("local", 1),
                make_opsetid("ai.onnx.ml", 2),
                make_opsetid("com.microsoft", 1),
                make_opsetid("onnx_extented.ortops.tutorial.cpu", 1),
            ],
            ir_version=9,
            functions=functions,
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(
        Version(ort_version) < Version("1.17.0"), reason="type inference failed"
    )
    def test_custom_gemm_local_function(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        for local in [False, True]:
            with self.subTest(use_local=local):
                onnx_model = self._get_model_dql(local)
                opts = SessionOptions()
                opts.register_custom_ops_library(get_ort_ext_libs()[0])
                try:
                    sess = InferenceSession(
                        onnx_model.SerializeToString(),
                        opts,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as e:
                    raise AssertionError(
                        f"Unable to create InferenceSession with "
                        f"onx={onnx_simple_text_plot(onnx_model)}"
                    ) from e
                self.assertNotEmpty(sess)


if __name__ == "__main__":
    # TestOrtOpTutorialCpu().test_custom_gemm_local_function()
    unittest.main(verbosity=2)
