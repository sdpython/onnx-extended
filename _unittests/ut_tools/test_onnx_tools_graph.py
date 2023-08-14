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
from onnx.checker import check_model
from onnx.defs import onnx_opset_version

try:
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime import __version__ as ort_version
except ImportError:
    InferenceSession, SessionOptions = None, None
    ort_version = "0.0"
if InferenceSession is not None:
    from onnxruntime import get_available_providers
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    ort_has_cuda = "CUDAExecutionProvider" in get_available_providers()
else:
    ort_has_cuda = False

from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.graph.onnx_graph_struct import Graph
from onnx_extended.tools.graph.onnx_graph_transformer import quantize_float8
from onnx_extended.tools.graph.onnx_custom_ops import GemmFloat8
from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs as get_ort_ext_libs_cpu
from onnx_extended import has_cuda

if has_cuda():
    from onnx_extended.validation.cuda.cuda_example_py import get_device_prop
else:
    get_device_prop = None


class TestOnnxToolsGraph(ExtTestCase):
    def _get_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        graph = make_graph(
            [
                make_node(
                    "Constant",
                    [],
                    ["one"],
                    value=make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
                ),
                make_node("Add", ["one", "one"], ["two"]),
                make_node("Add", ["X", "two"], ["xp"]),
                make_node("MatMul", ["X", "xp"], ["res"]),
                make_node("MatMul", ["X", "res"], ["Z"]),
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

    def test_graph_build(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 6)
        cst = []
        for node in graph:
            cst.append(node.is_constant())
        self.assertEqual([False, True, True, False, False, False], cst)

        ref = CReferenceEvaluator(model)
        x = np.random.random((3, 3)).astype(np.float32)
        z = ref.run(None, dict(X=x))[0]
        self.assertEqual(z.shape, (3, 3))
        self.assertEqualArray(x @ x @ (x + 2), z, atol=1e-5)
        self.assertEqual(len(list(graph)), 6)
        for i in range(0, 6):
            node = graph[i]
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul"})
        text = str(graph)
        tn = str(graph[1])
        self.assertEqual(tn, "Node(1, <parent>, <Constant>) [] -> [one]")
        self.assertEqual(text, "Graph(...) [X] -> [Z]")

        onx = graph.to_onnx()
        ref2 = CReferenceEvaluator(onx)
        got = ref2.run(None, dict(X=x))[0]
        self.assertEqualArray(z, got)

    def test_graph_build_initializer(self):
        onnx_model = make_model(
            make_graph(
                [make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])],
                "graph",
                [make_tensor_value_info("x", TensorProto.FLOAT, (None, None, None))],
                [make_tensor_value_info("y", TensorProto.FLOAT, (1, 6, 2))],
                initializer=[
                    make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                    make_tensor("ends", TensorProto.INT64, (2,), (2, 2)),
                    make_tensor("axes", TensorProto.INT64, (2,), (0, 2)),
                ],
            )
        )
        check_model(onnx_model)
        graph = Graph(onnx_model)
        self.assertEqual(len(graph), 5)
        for node in graph:
            self.assertEqual("Node(0, <parent>, <input>) [] -> [x]", str(node))
            break
        self.assertEqual("Graph(...) [x] -> [y]", str(graph))

        feeds = {"x": np.arange(4**3).reshape((-1, 4, 4)).astype(np.float32)}
        ref1 = CReferenceEvaluator(onnx_model)
        expected = ref1.run(None, feeds)[0]
        onx = graph.to_onnx()
        ref2 = CReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_graph_opsets(self):
        model = self._get_model()
        graph = Graph(model)
        opsets = graph.get_opsets()
        main = graph.get_opset()
        self.assertEqual(opsets[""], main)

    def test_graph_replace(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 6)
        node_set = graph.replace_nodes(3, make_node("Sub", ["X", "two"], ["xp"]))
        indices = [n.index for n in node_set]
        self.assertEqual(indices, [6])
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            if i == 3:
                self.assertRaise(lambda: graph[i], IndexError)
                continue
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual(indices, [0, 1, 2, 6, 4, 5])

        graph.simplify(False)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)

        graph.simplify(True)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(list(graph)), 6)
        ops = []
        for i in range(0, 6):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(node.op_type, {"input", "Constant", "Add", "MatMul", "Sub"})
        self.assertEqual(ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"])
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)

        onx = graph.to_onnx()
        ref2 = CReferenceEvaluator(onx)
        got = ref2.run(None, {"X": np.arange(9).reshape((-1, 3)).astype(np.float32)})[0]
        expected = np.array(
            [72, 126, 180, 234, 396, 558, 396, 666, 936], dtype=np.float32
        ).reshape((-1, 3))
        self.assertEqualArray(expected, got)

    def test_graph_remove(self):
        model = self._get_model()
        graph = Graph(model)
        self.assertEqual(len(graph), 6)
        graph.replace_nodes(3, make_node("Sub", ["X", "X"], ["xp"]))
        graph.simplify(False)
        removed = graph.remove_unused_nodes()
        self.assertEqual(len(removed), 2)
        self.assertEqual(str(removed[0]), "Node(2, <parent>, <Add>) [one,one] -> [two]")
        self.assertEqual(str(removed[1]), "Node(1, <parent>, <Constant>) [] -> [one]")

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
    def test_quantize_f8(self):
        model = self._get_model_32()
        graph = Graph(model)
        n_nodes = len(graph)
        new_graph = quantize_float8(graph)
        self.assertEqual(len(new_graph), len(graph))
        self.assertGreater(len(new_graph), n_nodes)

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
        check_model(onx2)
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
        check_model(onx2)
        try:
            ref2 = InferenceSession(
                onx2.SerializeToString(),
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
        self.assertIn("onnx_extented.ortops.tutorial.cpu", str(onx2))
        self.assertIn("onnx_extented.ortops.tutorial.cuda", str(onx2))

    @unittest.skipIf(
        not has_cuda() or not ort_has_cuda,
        reason="onnxruntime not installed or cuda is not available",
    )
    @unittest.skipIf(
        Version(ort_version) < Version("1.16"), reason="float8 types not released"
    )
    @unittest.skipIf(
        get_device_prop is None or get_device_prop().get("major") < 9,
        reason="Float 8 not supported on this machine",
    )
    def test_quantize_f8_onnx_extended(self):
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

    def test_quantize_f8_onnx_extended_code_local(self):
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

        new_graph = quantize_float8(graph, version="onnx-extended", local_function=True)
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("onnx_extented.ortops.tutorial.cuda", str(onx2))
        self.assertIn("local.quant.domain", str(onx2))

    @unittest.skipIf(onnx_opset_version() < 20, reason="onnx not recent enough")
    def test_quantize_f8_onnxruntime_code_local(self):
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

        new_graph = quantize_float8(graph, version="onnxruntime", local_function=True)
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("local.quant.domain", str(onx2))

        ref2 = CReferenceEvaluator(onx2, new_ops=[GemmFloat8])
        got2 = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got2, rtol=0.05)


if __name__ == "__main__":
    # import logging
    # logging.basicConfig(level=logging.ERROR)
    # log = logging.getLogger("onnx-extended")
    # log.setLevel(logging.ERROR)
    # TestOnnxToolsGraph().test_quantize_f8_onnx_extended()
    TestOnnxToolsGraph().test_quantize_f8_onnxruntime_code_local()
    unittest.main(verbosity=2)
