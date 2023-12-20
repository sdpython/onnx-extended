import unittest
import numpy
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
)
from onnx_extended.ext_test_case import ExtTestCase


class TestOrtOpOptimSparse(ExtTestCase):
    def test_dense_to_sparse(self):
        from onnxruntime import InferenceSession, SessionOptions
        from onnx_extended.validation.cpu._validation import sparse_struct_to_dense
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node = make_node(
            "DenseToSparse", ["X"], ["Y"], domain="onnx_extented.ortops.optim.cpu"
        )
        graph = make_graph([node], "g", [X], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("", 16),
                make_opsetid("onnx_extented.ortops.optim.cpu", 1),
            ],
        )
        check_model(onnx_model)
        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        x = numpy.arange(6).reshape((-1, 2)).astype(numpy.float32)
        got = sess.run(None, {"X": x})
        back = sparse_struct_to_dense(got[0])
        self.assertEqualArray(x, back)


if __name__ == "__main__":
    unittest.main(verbosity=2)
