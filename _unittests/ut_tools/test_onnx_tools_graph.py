import itertools
import unittest
import numpy as np
from packaging.version import Version
from onnx import TensorProto, __version__ as onnx_version
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
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear
from onnx.reference.op_run import to_array_extended
from onnx.onnx_cpp2py_export.defs import SchemaError

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
from onnx_extended.helper import make_dynamic_quantize_linear_function_proto
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.graph.onnx_graph_struct import Graph
from onnx_extended.tools.graph.onnx_graph_transformer import (
    cast_constant,
    quantize_float8,
    QuantizationError,
)
from onnx_extended.tools.graph.onnx_custom_ops import GemmFloat8, GemmFloat8Quiet
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
    def _get_basic_square_model(self, init, n_dim_x, n_dim_c, side_x):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None] * n_dim_x)
        Y = make_tensor_value_info(
            "Y", TensorProto.FLOAT, [None] * max(n_dim_x, n_dim_c)
        )
        shape_cst = np.array([2] * n_dim_c).astype(np.int64)
        value_cst = (np.arange(np.prod(shape_cst)) / np.prod(shape_cst)).astype(
            np.float32
        )
        matmul = make_node(
            "MatMul", ["X", "cst"] if side_x == 0 else ["cst", "X"], ["Y"]
        )
        cst = make_tensor(
            "cst", TensorProto.FLOAT, shape_cst.tolist(), value_cst.tolist()
        )
        if init:
            nodes = [matmul]
            inits = [cst]
        else:
            nodes = [
                make_node("Constant", [], ["cst"], value=cst),
                matmul,
            ]
            inits = []

        graph = make_graph(nodes, "zoo", [X], [Y], inits)
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        return onnx_model, value_cst.reshape(tuple(shape_cst.tolist()))

    def test_basic_all(self):
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        sess_opts = SessionOptions()
        sess_opts.register_custom_ops_library(get_ort_ext_libs()[0])

        # Let's create a function equivalent to DynamicQuantizeLinear
        dynql = ReferenceEvaluator(
            make_dynamic_quantize_linear_function_proto(domain="qtest", opset=18)
        )
        atts = dict(to=TensorProto.FLOAT8E4M3FN)

        def dynamic_qdq_linear(x):
            qx, scale, _ = dynql.run(None, dict(x=x), attributes=atts)
            qdq = DequantizeLinear.eval(qx, scale)
            return qdq

        def check_onx(onx, tr):
            # check transpose are correct
            for init in onx.graph.initializer:
                if init.name.startswith("cst") and "scale" not in init.name:
                    value = to_array_extended(init)
                    if value.dtype == np.float32:
                        raise AssertionError(
                            f"Iniatialier {init.name!r} "
                            f"has dtype {value.dtype}: {init}."
                        )
            for node in onx.graph.node:
                if node.op_type not in {"GemmFloat8", "CustomGemmFloat8E4M3FN"}:
                    continue
                for att in node.attribute:
                    if att.name == "transA":
                        if att.i != (1 if tr & 1 else 0):
                            raise AssertionError(
                                f"Unexpected value for transA in\n-----\n"
                                f"{onnx_simple_text_plot(onx)}"
                            )
                    elif att.name == "transB":
                        if att.i != (1 if tr & 2 else 0):
                            raise AssertionError(
                                f"Unexpected value for transB in\n-----\n"
                                f"{onnx_simple_text_plot(onx)}"
                            )

        options = itertools.product(
            [True, False],  # init
            [0, 1, 2, 3],  # tr
            [0, 1],  # side_
            [2, 3],  # n_dim_x
            [3, 2],  # n_dim_c
        )

        for it, (init, tr, side_x, n_dim_x, n_dim_c) in enumerate(options):
            msg = (
                f"init={init}, tr={tr}, side_x={side_x}, "
                f"n_dim_x={n_dim_x}, n_dim_c={n_dim_c}"
            )
            with self.subTest(msg=msg):
                # print(f"-----------------------------\n{msg}")
                model, cst = self._get_basic_square_model(
                    init=init, n_dim_x=n_dim_x, n_dim_c=n_dim_c, side_x=side_x
                )

                x = np.random.random((2,) * n_dim_x).astype(np.float32)
                feeds = dict(X=x)

                try:
                    ref = CReferenceEvaluator(model)
                except RuntimeError as e:
                    raise AssertionError(
                        f"Unable to load model\n----\n{onnx_simple_text_plot(model)}"
                    ) from e
                z0 = ref.run(None, feeds)[0]

                # Let's compute expected value after quandization
                qx, qc = dynamic_qdq_linear(x), dynamic_qdq_linear(cst)
                expected = qx @ qc if side_x == 0 else qc @ qx
                self.assertEqualArray(expected, z0, atol=0.5)

                graph = Graph(model)
                try:
                    new_graph = quantize_float8(graph, index_transpose=tr)
                except QuantizationError as e:
                    if n_dim_x > 2 and n_dim_c > 2:
                        continue
                    raise e
                onx = new_graph.to_onnx()
                check_onx(onx, tr)

                # check the reshape is there if the dimension is greather than 3
                if max(n_dim_x, n_dim_c) > 2 and (
                    'op_type: "Reshape"' not in str(onx) or len(onx.functions) <= 1
                ):
                    raise AssertionError(
                        f"Dimension is 3 but Reshape is missing "
                        f"or the number of functions is <= 1 in "
                        f"\n----\n{msg}\n{onnx_simple_text_plot(onx)}"
                    )

                # let's replace the operator by another one not checking
                # transA and transB attributes during execution
                for node in onx.graph.node:
                    if node.op_type == "GemmFloat8":
                        node.op_type = "GemmFloat8Quiet"
                try:
                    ref2 = CReferenceEvaluator(onx, new_ops=[GemmFloat8Quiet])
                except RuntimeError as e:
                    raise AssertionError(
                        f"Unable to load model\n----\n{msg}\n"
                        f"{onnx_simple_text_plot(onx)}"
                    ) from e
                try:
                    got = ref2.run(None, dict(X=x))[0]
                except SchemaError as es:
                    if Version(onnx_version) < Version("1.16.0") and (
                        "No schema registered for 'Cast_19'" in str(es)
                    ):
                        continue
                    raise es
                except (ValueError, RuntimeError) as e:
                    if Version(onnx_version) < Version("1.16.0") and (
                        "Both types must be float 8" in str(e)
                    ):
                        continue
                    raise AssertionError(
                        f"Unable to run model with x.shape={x.shape}"
                        f"\n----\n{msg}\n{onnx_simple_text_plot(onx)}"
                    ) from e
                try:
                    self.assertEqualArray(expected, got, atol=1e-5)
                except AssertionError as e:
                    raise AssertionError(
                        f"Verification failed with GemmFloat8Quiet\n"
                        f"expected.shape={expected.shape} got.shape={got.shape}\n"
                        f"x=\n{x}\nqx=\n{qx}\ncst=\n{cst}\nqc=\n{qc}\n--\n"
                        f"expected=\n{expected}\ngot={got}\n----\n{msg}\n"
                        f"onx={onnx_simple_text_plot(onx)}"
                    ) from e

                # check with onnxruntime and CPU kernel
                graph = Graph(model)
                new_graph = quantize_float8(
                    graph,
                    version="onnx-extended",
                    domain_ops={
                        "CustomGemmFloat8E4M3FN": "onnx_extented.ortops.tutorial.cpu"
                    },
                    index_transpose=tr,
                )
                onxo = new_graph.to_onnx()
                check_onx(onxo, tr)
                try:
                    sess = InferenceSession(
                        onxo.SerializeToString(),
                        sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                except OrtFail as e:
                    if "type inference failed" in str(e):
                        # bug of onnxruntime
                        with open(
                            f"custom_ops_type_inference_fails_{it}.onnx", "wb"
                        ) as f:
                            f.write(onxo.SerializeToString())
                        continue
                    raise e
                got = sess.run(None, dict(X=x))[0]
                self.assertEqualArray(expected, got, atol=1e-5)

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
                make_node("MatMul", ["X", "xp"], ["res"], name="m1"),
                make_node("MatMul", ["X", "res"], ["Z"], name="m2"),
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
        self.assertEqual(len(graph), 7)
        cst = []
        for node in graph:
            cst.append(node.is_constant())
        self.assertEqual([False, True, True, False, False, False, False], cst)

        ref = CReferenceEvaluator(model)
        x = np.random.random((3, 3)).astype(np.float32)
        z = ref.run(None, dict(X=x))[0]
        self.assertEqual(z.shape, (3, 3))
        self.assertEqualArray(x @ x @ (x + 2), z, atol=1e-5)
        self.assertEqual(len(list(graph)), 7)
        for i in range(0, 7):
            node = graph[i]
            self.assertIn(
                node.op_type, {"input", "output", "Constant", "Add", "MatMul"}
            )
        text = str(graph)
        tn = str(graph[1])
        self.assertEqual(tn, "Node(1, <parent>, <Constant>) [1:(1,)] -> [one]")
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
        self.assertEqual(len(graph), 6)
        for node in graph:
            self.assertEqual(
                "Node(0, <parent>, kind=NodeKind.INPUT) [1:('', '', '')] -> [x]",
                str(node),
            )
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
        self.assertEqual(len(graph), 7)
        node_set = graph.replace_nodes(3, make_node("Sub", ["X", "two"], ["xp"]))
        indices = [n.index for n in node_set]
        self.assertEqual(indices, [7])
        self.assertEqual(len(graph), 7)
        self.assertEqual(len(list(graph)), 7)
        ops = []
        for i in range(0, 7):
            if i == 3:
                self.assertRaise(lambda: graph[i], IndexError)
                continue
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(
                node.op_type, {"input", "output", "Constant", "Add", "MatMul", "Sub"}
            )
        self.assertEqual(
            ops, ["input", "Constant", "Add", "MatMul", "MatMul", "output"]
        )
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul", "output"]
        )
        indices = [node.index for node in graph]
        self.assertEqual(indices, [0, 1, 2, 7, 4, 5, 6])

        graph.simplify(False)
        self.assertEqual(len(graph), 7)
        self.assertEqual(len(list(graph)), 7)
        ops = []
        for i in range(0, len(graph)):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(
                node.op_type, {"input", "output", "Constant", "Add", "MatMul", "Sub"}
            )
        self.assertEqual(
            ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul", "output"]
        )
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul", "output"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5, 6], indices)

        graph.simplify(True)
        self.assertEqual(len(graph), 7)
        self.assertEqual(len(list(graph)), 7)
        ops = []
        for i in range(0, len(graph)):
            node = graph[i]
            ops.append(node.op_type)
            self.assertIn(
                node.op_type, {"input", "output", "Constant", "Add", "MatMul", "Sub"}
            )
        self.assertEqual(
            ops, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul", "output"]
        )
        op_types = [node.op_type for node in graph]
        self.assertEqual(
            op_types, ["input", "Constant", "Add", "Sub", "MatMul", "MatMul", "output"]
        )
        indices = [node.index for node in graph]
        self.assertEqual([0, 1, 2, 3, 4, 5, 6], indices)

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
        self.assertEqual(len(graph), 7)
        graph.replace_nodes(3, make_node("Sub", ["X", "X"], ["xp"]))
        graph.simplify(False)
        removed = graph.remove_unused_nodes()
        self.assertEqual(len(removed), 2)
        self.assertEqual(str(removed[0]), "Node(2, <parent>, <Add>) [one,one] -> [two]")
        self.assertEqual(
            str(removed[1]), "Node(1, <parent>, <Constant>) [1:(1,)] -> [one]"
        )

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

    def test_cast_constant_constant(self):
        x32 = np.arange(24 * 5).reshape((5, 2, 4, 3)).astype(np.float32)
        x16 = x32.astype(np.float16)
        feeds32 = {"X": x32}
        feeds16 = {"X": x16}
        model = self._get_model_32_x4()
        refonnx = CReferenceEvaluator(model)
        expected = refonnx.run(None, feeds32)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds32)[0]
        self.assertEqualArray(expected, got1)

        new_graph = cast_constant(
            graph, from_type=TensorProto.FLOAT, to_type=TensorProto.FLOAT16
        )
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("data_type: 10", str(onx2))

        ref2 = CReferenceEvaluator(onx2, verbose=10)
        got2 = ref2.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got2, rtol=0.05)

        sess = InferenceSession(
            onx2.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got3 = sess.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got3, rtol=0.05)

    def test_cast_constant_initializer(self):
        x32 = np.arange(24 * 5).reshape((5, 2, 4, 3)).astype(np.float32)
        x16 = x32.astype(np.float16)
        feeds32 = {"X": x32}
        feeds16 = {"X": x16}
        model = self._get_model_32_x4(use_init=True)
        refonnx = CReferenceEvaluator(model)
        expected = refonnx.run(None, feeds32)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds32)[0]
        self.assertEqualArray(expected, got1)

        new_graph = cast_constant(
            graph, from_type=TensorProto.FLOAT, to_type=TensorProto.FLOAT16
        )
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("data_type: 10", str(onx2))

        ref2 = CReferenceEvaluator(onx2)
        got2 = ref2.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got2, rtol=0.05)

        sess = InferenceSession(
            onx2.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got3 = sess.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got3, rtol=0.05)

    def _get_model_32_x4_cast(self, use_init=False):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [5, 2, 4, 3])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None, None, None])
        if use_init:
            graph = make_graph(
                [
                    make_node("Cast", ["X"], ["Xc"], to=TensorProto.FLOAT),
                    make_node("MatMul", ["Xc", "mat"], ["Z"]),
                ],
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
                    make_node("Cast", ["X"], ["Xc"], to=TensorProto.FLOAT),
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
                    make_node("MatMul", ["Xc", "mat"], ["Z"]),
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

    def test_cast_constant_initializer_cast(self):
        x32 = np.arange(24 * 5).reshape((5, 2, 4, 3)).astype(np.float32)
        x16 = x32.astype(np.float16)
        feeds32 = {"X": x32}
        feeds16 = {"X": x16}
        model = self._get_model_32_x4_cast(use_init=True)
        refonnx = CReferenceEvaluator(model)
        expected = refonnx.run(None, feeds32)[0]

        graph = Graph(model)
        onx1 = graph.to_onnx()
        check_model(onx1)
        ref1 = CReferenceEvaluator(onx1)
        got1 = ref1.run(None, feeds32)[0]
        self.assertEqualArray(expected, got1)

        new_graph = cast_constant(
            graph, from_type=TensorProto.FLOAT, to_type=TensorProto.FLOAT16
        )
        onx2 = new_graph.to_onnx()
        check_model(onx2)
        self.assertIn("data_type: 10", str(onx2))

        ref2 = CReferenceEvaluator(onx2)
        got2 = ref2.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got2, rtol=0.05)

        sess = InferenceSession(
            onx2.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got3 = sess.run(None, feeds16)[0]
        self.assertEqualArray(expected.astype(np.float16), got3, rtol=0.05)


if __name__ == "__main__":
    import logging

    for name in ["onnx-extended", "skl2onnx"]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    TestOnnxToolsGraph().test_basic_all()
    # TestOnnxToolsGraph().test_quantize_f8_onnx_onnxruntime()
    unittest.main(verbosity=2)
