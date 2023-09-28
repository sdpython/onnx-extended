import unittest
import numpy as np
from packaging.version import Version
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model, ValidationError
from onnx.defs import onnx_opset_version

try:
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime import __version__ as ort_version
except ImportError:
    InferenceSession, SessionOptions = None, None
    ort_version = "0.0"
if InferenceSession is not None:
    from onnxruntime import get_available_providers
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        InvalidArgument,
        Fail as OrtFail,
    )

    ort_has_cuda = "CUDAExecutionProvider" in get_available_providers()
else:
    ort_has_cuda = False

try:
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
except ImportError:
    onnx_simple_text_plot = str

from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.graph.onnx_graph_struct import Graph
from onnx_extended.tools.graph.onnx_graph_transformer import (
    quantize_float8,
    QuantizeOptions,
)
from onnx_extended.tools.graph.onnx_custom_ops import GemmFloat8
from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs as get_ort_ext_libs_cpu
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import get_device_prop

    try:
        device_props = get_device_prop()
    except RuntimeError:
        device_props = {}
else:
    device_props = {}


class TestOnnxToolsGraph(ExtTestCase):
    def _get_model_32(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["mat"],
                    value=make_tensor(
                        "one",
                        TensorProto.FLOAT,
                        [3, 2],
                        list(float(i) for i in range(11, 17)),
                    ),
                ),
                make_node("MatMul", ["X", "mat"], ["Z"], name="m1"),
            ],
            "zoo",
            [X],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8(self):
        model = self._get_model_32()
        graph = Graph(model)
        n_nodes = len(graph)
        new_graph = quantize_float8(graph)
        self.assertEqual(len(new_graph), len(graph))
        self.assertGreater(len(new_graph), n_nodes)

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_exceptions(self):
        model = self._get_model_32()
        graph = Graph(model)
        new_graph = quantize_float8(graph, exceptions=[dict(name="m1")])
        self.assertEmpty(new_graph)

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnx(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = CReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph)
        onx2 = new_graph.to_onnx()
        try:
            check_model(onx2)
        except ValidationError as e:
            if (
                "Bad node spec for node. Name: dql8_X OpType: DynamicQuantizeLinear"
                in str(e)
            ):
                # onnx not recent enough
                return
            raise e

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnx_optimize(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = CReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, quantize_options=QuantizeOptions.OPTIMIZE)
        onx2 = new_graph.to_onnx()
        try:
            check_model(onx2)
        except ValidationError as e:
            if (
                "Bad node spec for node. Name: dql8_X OpType: DynamicQuantizeLinear"
                in str(e)
            ):
                # onnx not recent enough
                return
            raise e

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)

    @unittest.skipIf(
        onnx_opset_version() < 20 and (not has_cuda() or not ort_has_cuda),
        reason="onnx not recent enough or onnxruntime not "
        "installed or cuda is not available",
    )
    def test_quantize_f8_onnx_onnxruntime(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph)
        onx2 = new_graph.to_onnx()
        try:
            check_model(onx2)
        except ValidationError as e:
            if (
                "Bad node spec for node. Name: dql8_X OpType: DynamicQuantizeLinear"
                in str(e)
            ):
                # onnx not recent enough
                return
            raise e
        try:
            ref2 = InferenceSession(
                onx2.SerializeToString(),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except OrtFail as e:
            raise AssertionError(
                f"Unable to load model\n----\n{onnx_simple_text_plot(onx2)}"
            ) from e
        except InvalidArgument as e:
            if "Current official support for domain ai.onnx is till opset 19." in str(
                e
            ):
                # onnxruntime not recent enough
                return
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)

    @unittest.skipIf(
        not has_cuda() or not ort_has_cuda,
        reason="onnxruntime not installed or cuda is not available",
    )
    def test_quantize_f8_onnx_extended_code(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnx-extended")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("onnx_extented.ortops.tutorial.cuda", str(onx2))

    @unittest.skipIf(
        not has_cuda() or not ort_has_cuda,
        reason="onnxruntime not installed or cuda is not available",
    )
    @unittest.skipIf(
        Version(ort_version) < Version("1.16"), reason="float8 types not released"
    )
    @unittest.skipIf(
        device_props.get("major", 0) < 9,
        reason="Float 8 not supported on this machine",
    )
    def test_quantize_f8_onnx_extended_cuda(self):
        from onnx_extended.ortops.tutorial.cuda import (
            get_ort_ext_libs as get_ort_ext_libs_cuda,
        )

        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnx-extended")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("onnx_extented.ortops.tutorial.cpu", str(onx2))
        self.assertIn("onnx_extented.ortops.tutorial.cuda", str(onx2))

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])
        r = get_ort_ext_libs_cuda()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        try:
            ref2 = InferenceSession(
                onx2.SerializeToString(),
                opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except InvalidArgument as e:
            if "Current official support for domain ai.onnx is till opset 19." in str(
                e
            ):
                # onnxruntime not recent enough
                return
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)

    def _test_quantize_f8_onnx_extended_cpu(self):
        from onnx_extended.ortops.tutorial.cpu import (
            get_ort_ext_libs as get_ort_ext_libs_cpu,
        )

        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(
            graph,
            version="onnx-extended",
            domain_ops={"CustomGemmFloat8E4M3FN": "onnx_extented.ortops.tutorial.cpu"},
        )
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("onnx_extented.ortops.tutorial.cpu", str(onx2))
        self.assertNotIn("onnx_extented.ortops.tutorial.cuda", str(onx2))

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        try:
            ref2 = InferenceSession(
                onx2.SerializeToString(),
                opts,
                providers=["CPUExecutionProvider"],
            )
        except InvalidArgument as e:
            if "Current official support for domain ai.onnx is till opset 19." in str(
                e
            ):
                # onnxruntime not recent enough
                return
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)

    def test_quantize_f8_onnx_extended_cpu_cuda(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnx-extended")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))
        self.assertIn("onnx_extented.ortops.tutorial.cuda", str(onx2))

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnxruntime(self):
        x = np.arange(12).reshape((4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32()
        ref = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = ref.run(None, feeds)[0]

        opts = SessionOptions()
        r = get_ort_ext_libs_cpu()
        self.assertNotEmpty(r)
        opts.register_custom_ops_library(r[0])

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = InferenceSession(
            onx1.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnxruntime")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        try:
            got2 = ref2.run(None, feeds)[0]
        except ValueError as e:
            raise AssertionError(
                f"Unable to run model\n----\n" f"{onnx_simple_text_plot(onx2)}\n------"
            ) from e
        self.assertEqualArray(expected, got2, rtol=0.05)

    def _get_model_32_x3(self, transpose=False):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["mat"],
                    value=make_tensor(
                        "one",
                        TensorProto.FLOAT,
                        [2, 3] if transpose else [3, 2],
                        list(float(i) for i in range(11, 17)),
                    ),
                ),
                make_node("MatMul", ["mat", "X"] if transpose else ["X", "mat"], ["Z"]),
            ],
            "zoo",
            [X],
            [Z],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnxruntime_x3(self):
        x = np.arange(24).reshape((2, 4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32_x3()
        refonnx = CReferenceEvaluator(model)
        try:
            expected = refonnx.run(None, feeds)[0]
        except (ValueError, TypeError) as e:
            raise AssertionError(
                f"Unable to run model\n---\n{onnx_simple_text_plot(model)}"
            ) from e

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnxruntime")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        try:
            got2 = ref2.run(None, feeds)[0]
        except ValueError as e:
            raise AssertionError(
                f"Unable to run model\n---\n{onnx_simple_text_plot(onx2)}"
            ) from e
        self.assertEqualArray(expected, got2, rtol=0.05)

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnxruntime_x3_transpose(self):
        x = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32_x3(transpose=True)
        refonnx = CReferenceEvaluator(model)
        try:
            expected = refonnx.run(None, feeds)[0]
        except (ValueError, TypeError) as e:
            raise AssertionError(
                f"Unable to run model\n---\n{onnx_simple_text_plot(model)}"
            ) from e

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnxruntime")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        try:
            got2 = ref2.run(None, feeds)[0]
        except ValueError as e:
            raise AssertionError(
                f"Unable to run model\n---\n{onnx_simple_text_plot(onx2)}"
            ) from e
        self.assertEqualArray(expected, got2, rtol=0.05)

    def _get_model_32_x4(self, use_init=False):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [5, 2, 4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None, None])
        if use_init:
            graph = make_graph(
                [make_node("MatMul", ["X", "mat"], ["Z"])],
                "zoo",
                [X],
                [Z],
                [
                    make_tensor(
                        "mat",
                        TensorProto.FLOAT,
                        [3, 2],
                        list(float(i) for i in range(11, 17)),
                    )
                ],
            )
        else:
            graph = make_graph(
                [
                    make_node(
                        "Constant",
                        [],
                        ["mat"],
                        value=make_tensor(
                            "one",
                            TensorProto.FLOAT,
                            [3, 2],
                            list(float(i) for i in range(11, 17)),
                        ),
                    ),
                    make_node("MatMul", ["X", "mat"], ["Z"]),
                ],
                "zoo",
                [X],
                [Z],
            )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnxruntime_x4(self):
        x = np.arange(24 * 5).reshape((5, 2, 4, 3)).astype(np.float32)
        feeds = {"X": x}
        model = self._get_model_32_x4()
        refonnx = CReferenceEvaluator(model)
        expected = refonnx.run(None, feeds)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds)[0]
        self.assertEqualArray(expected, got1)

        new_graph = quantize_float8(graph, version="onnxruntime")
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)


if __name__ == "__main__":
    import logging

    for name in ["onnx-extended", "skl2onnx"]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    TestOnnxToolsGraph().test_quantize_f8_onnx_optimize()
    unittest.main(verbosity=2)
