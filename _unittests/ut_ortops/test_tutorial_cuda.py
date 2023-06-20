import unittest
import numpy
from itertools import product
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
from onnx_extended.ortops.tutorial.cuda import documentation
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import get_device_prop
else:
    get_device_prop = None


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

    def common_test_custom_gemm(self, op_name, tos, return_sess=False, **kwargs):
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
            node_inputs += ["scaleA", "scaleB", "scaleY"]
        nodes = [
            *casts,
            make_node(
                op_name,
                node_inputs,
                node_outputs,
                domain="onnx_extented.ortops.tutorial.cuda",
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
                    make_tensor_value_info("scaleY", TensorProto.FLOAT, [1]),
                ]
            )
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
        if return_sess:
            return onnx_model, sess

        inputs = [
            (numpy.arange(256) / 256).astype(numpy.float32).reshape((-1, 16))
            for to in tos
        ]
        feeds = dict(zip("AB", inputs))
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
        a, b = inputs[:2]
        if kwargs.get("rowMajor", 1):
            expected = a.T @ b
        else:
            expected = a @ b.T
        expected *= kwargs.get("alpha", 1.0)
        self.assertEqualArray(expected, got[0], atol=0.08 if gemm8 else 1e-6)

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
            transA=1,
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
            transA=1,
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
    @unittest.skipIf(
        get_device_prop is None or get_device_prop().get("major") < 9,
        reason="Float 8 not supported on this machine",
    )
    def test_custom_gemm_float8(self):
        self.common_test_custom_gemm(
            "CustomGemmFloat8E4M3FN",
            [TensorProto.FLOAT8E4M3FN for i in range(2)],
            name="cgf8",
            transA=1,
            fastAccumulationMode=1,
            rowMajor=0,
        )

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @unittest.skipIf(
        "CUDAExecutionProvider" not in get_available_providers(),
        reason="CUDA provider not available",
    )
    def test_custom_gemm_all_possible(self):
        excs = []
        booleans = [0, 1]
        dims = [9, 12]
        shapes = [
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(3, 3), (3, 3)],
            [(4, 3), (3, 4)],
            [(8, 3), (3, 4)],
            [(12, 3), (3, 4)],
            [(16, 3), (3, 4)],
            [(4, 3), (4, 3)],
            [(8, 3), (4, 3)],
            [(12, 3), (4, 3)],
            [(16, 3), (4, 3)],
            [(4, 3), (4, 3)],
            [(12, 3), (12, 1)],
            [(4, 3), (3, 4)],
            [(4, 3), (3, 4)],
            [(8, 3), (3, 4)],
            [(12, 3), (3, 4)],
            [(16, 3), (3, 4)],
            [(4, 3), (4, 3)],
            [(8, 3), (4, 3)],
            [(12, 3), (4, 3)],
            [(16, 3), (4, 3)],
            [(4, 3), (4, 3)],
            [(12, 3), (12, 1)],
            [(4, 3), (3, 4)],
        ]

        for N, rm, transa, transb, sh in product(
            dims, booleans, booleans, booleans, shapes
        ):
            if len(excs) > 1:
                # too many errors
                break
            row_major = 1 - rm
            order = "C" if row_major else "F"

            sha, shb = sh
            a = (
                numpy.arange(numpy.prod(sha))
                .reshape(sha, order=order)
                .astype(numpy.float32)
            )
            b = (numpy.arange(numpy.prod(shb)).reshape(shb, order=order) * 10).astype(
                numpy.float32
            )
            shapes = [a.shape, b.shape]

            with self.subTest(
                transa=transa,
                transB=transb,
                rowMajor=row_major,
                sh1=a.shape,
                sh2=b.shape,
            ):
                if not row_major:
                    # onnxruntime does not take into account the storage.
                    # it is always row major.
                    # A matrix RxC column major is equal to A.T

                    if a.shape[0] == a.shape[1] and b.shape[0] == b.shape[1]:
                        am = a.T.copy()
                        bm = b.T.copy()
                    else:
                        am = a.T.copy().reshape(a.shape)
                        bm = b.T.copy().reshape(b.shape)
                    at = am.T if transa else am
                    bt = bm.T if transb else bm

                    try:
                        expected = (at @ bt).T
                    except ValueError:
                        # Not possible
                        continue
                else:
                    at = a.T if transa else a
                    bt = b.T if transb else b

                    try:
                        expected = at @ bt
                    except ValueError:
                        # Not possible
                        continue

                onx, sess = self.common_test_custom_gemm(
                    "CustomGemmFloat",
                    [TensorProto.FLOAT for i in range(2)],
                    name="cgf",
                    fastAccumulationMode=1,
                    computeType="CUBLAS_COMPUTE_32F_FAST_TF32",
                    transA=transa,
                    transB=transb,
                    rowMajor=row_major,
                    return_sess=True,
                )

                feeds = {"A": a, "B": b}
                try:
                    got = sess.run(None, feeds)[0]
                except Exception as e:
                    excs.append(("A", N, row_major, transa, transb, sh))
                    raise AssertionError(
                        f"Unable to execute model with a.shape={a.shape}, "
                        f"b.shape={b.shape} and row_major={row_major}."
                        f"\n{onnx_simple_text_plot(onx)}."
                    ) from e
                try:
                    self.assertEqualArray(expected, got)
                except AssertionError as e:
                    strn = (  # noqa: E731
                        lambda s: str(s)
                        .replace("\n", " ")
                        .replace("  ", " ")
                        .replace(" : ", ":")
                    )
                    excs.append(("B", N, row_major, transa, transb, sh))
                    raise AssertionError(
                        f"row_major={row_major}, transa={transa}, transb={transb}, "
                        f"\na.shape={a.shape},\na.flags={strn(a.flags)}, "
                        f"\nb.shape={b.shape},\nb.flags={strn(b.flags)}, "
                        f"\nexpected.shape={expected.shape},"
                        f"\nexpected.flags={strn(expected.flags)}, "
                        f"\na=\n{a}\nb=\n{b}\n"
                    ) from e


if __name__ == "__main__":
    # TestOrtOpTutorialCuda().test_custom_gemm_all_possible()
    unittest.main(verbosity=2)
