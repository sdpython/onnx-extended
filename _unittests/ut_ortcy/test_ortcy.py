import unittest
import warnings
import os
import sys
import numpy
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnx_extended.ext_test_case import ExtTestCase

try:
    from onnx_extended.ortcy.wrap.ortinf import (
        OrtSession,
        get_ort_c_api_supported_version,
    )
except ImportError as e:
    try:
        import onnx_extended.ortcy.wrap.ortinf

        path = onnx_extended.ortcy.wrap.ortinf.__file__
    except ImportError as ee:
        path = str(ee)
    msg = "libonnxruntime.so.1.22.0: cannot open shared object file"
    if msg in str(e):
        from onnx_extended.ortcy.wrap import __file__ as loc

        all_files = os.listdir(os.path.dirname(loc))
        warnings.warn(
            f"Unable to find onnxruntime {e!r}, found files in {os.path.dirname(loc)}: "
            f"{all_files}, path={path!r}.",
            stacklevel=0,
        )
        OrtSession = None
        get_ort_c_api_supported_version = None
        here = os.path.dirname(__file__)
    else:
        import onnx_extended.ortcy.wrap as wp

        OrtSession = (
            f"OrtSession is not initialized:\n{e}"
            f"\n--found--\n{os.listdir(os.path.dirname(wp.__file__))}"
        )
        get_ort_c_api_supported_version = (
            f"get_ort_c_api_supported_version is not initialized:\n{e}"
            f"\n--found--\n{os.listdir(os.path.dirname(wp.__file__))}"
        )


class TestOrtCy(ExtTestCase):

    def test_get_ort_c_api_supported_version_str(self):
        assert not isinstance(get_ort_c_api_supported_version, str), (
            f"Unexpected value for get_ort_c_api_supported_version="
            f"{get_ort_c_api_supported_version!r}"
        )

    @unittest.skipIf(
        get_ort_c_api_supported_version is None
        or isinstance(get_ort_c_api_supported_version, str),
        reason="libonnxruntime installation failed",
    )
    def test_get_ort_c_api_supported_version(self):
        v = get_ort_c_api_supported_version()
        self.assertGreaterEqual(v, 16)

    @unittest.skipIf(
        OrtSession is None or isinstance(OrtSession, str),
        reason="libonnxruntime installation failed",
    )
    def test_ort_get_available_providers(self):
        from onnx_extended.ortcy.wrap.ortinf import ort_get_available_providers

        res = ort_get_available_providers()
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 0)
        self.assertIn("CPUExecutionProvider", res)

    @unittest.skipIf(
        OrtSession is None or isinstance(OrtSession, str),
        reason="libonnxruntime installation failed",
    )
    @unittest.skipIf(
        sys.platform == "darwin",
        reason="The test is unstable and leads to a crash `Illegal instruction`. "
        "It is probably due to the fact all machines used on CI are not the same. "
        "Some old machines do not recognized newer instructions.",
    )
    def test_session(self):
        from onnx_extended.ortcy.wrap.ortinf import OrtSession

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None, None])
        node = make_node("Add", ["X", "Y"], ["Z"])
        graph = make_graph([node], "add", [X, Y], [Z])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 18)], ir_version=8
        )
        check_model(onnx_model)
        data = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data):
            os.mkdir(data)
        name = os.path.join(data, "add.onnx")
        if not os.path.exists(name):
            with open(name, "wb") as f:
                f.write(onnx_model.SerializeToString())
        self.assertExists(name)

        session = OrtSession(name)
        self.assertEqual(session.get_input_count(), 2)
        self.assertEqual(session.get_output_count(), 1)

        data = onnx_model.SerializeToString()
        self.assertIsInstance(data, bytes)
        session = OrtSession(data)
        self.assertEqual(session.get_input_count(), 2)
        self.assertEqual(session.get_output_count(), 1)

        x = numpy.random.randn(2, 3).astype(numpy.float32)
        y = numpy.random.randn(2, 3).astype(numpy.float32)
        got = session.run_2(x, y)
        self.assertIsInstance(got, list)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(got[0], x + y)

        got = session.run([x, y])
        self.assertIsInstance(got, list)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(got[0], x + y)

    @unittest.skipIf(
        OrtSession is None or isinstance(OrtSession, str),
        reason="libonnxruntime installation failed",
    )
    @unittest.skipIf(
        sys.platform == "darwin", reason="Compilation settings fails on darwin"
    )
    def test_my_custom_ops_cy(self):
        from onnx_extended.ortcy.wrap.ortinf import OrtSession
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "MyCustomOp", ["X", "A"], ["Y"], domain="onnx_extended.ortops.tutorial.cpu"
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[make_opsetid("onnx_extended.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        session = OrtSession(
            onnx_model.SerializeToString(), custom_libs=get_ort_ext_libs()
        )
        self.assertEqual(session.get_input_count(), 2)
        self.assertEqual(session.get_output_count(), 1)

        x = numpy.random.randn(2, 3).astype(numpy.float32)
        y = numpy.random.randn(2, 3).astype(numpy.float32)
        got = session.run_2(x, y)[0]
        self.assertEqualArray(x + y, got)

    @unittest.skipIf(
        OrtSession is None or isinstance(OrtSession, str),
        reason="libonnxruntime installation failed",
    )
    @unittest.skipIf(
        sys.platform == "darwin", reason="Compilation settings fails on darwin"
    )
    def test_my_custom_ops_with_attributes(self):
        from onnx_extended.ortcy.wrap.ortinf import OrtSession
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.DOUBLE, [None, None])
        A = make_tensor_value_info("A", TensorProto.DOUBLE, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.DOUBLE, [None, None])
        node1 = make_node(
            "MyCustomOpWithAttributes",
            ["X", "A"],
            ["Y"],
            domain="onnx_extended.ortops.tutorial.cpu",
            att_string="string_att",
            att_int64=5,
            att_float=4.5,
            att_tensor=from_array(numpy.array([[5.1]], dtype=numpy.float64)),
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[make_opsetid("onnx_extended.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        session = OrtSession(
            onnx_model.SerializeToString(), custom_libs=get_ort_ext_libs()
        )
        self.assertEqual(session.get_input_count(), 2)
        self.assertEqual(session.get_output_count(), 1)

        x = numpy.random.randn(2, 3).astype(numpy.float64)
        y = numpy.random.randn(2, 3).astype(numpy.float64)
        got = session.run_2(x, y)[0]
        cst = 5.1 + 4.5 + 5 + ord("s")
        self.assertEqualArray(x + y + cst, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
