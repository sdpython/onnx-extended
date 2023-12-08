import unittest
import numpy
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnx_extended.ortops.optim.cpu import documentation
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
)
from onnx_extended.ext_test_case import ExtTestCase

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None


class TestOrtOpOptimTreeEnsembleSparseCpu(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 5)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)
        self.assertIn("Sparse", "\n".join(doc))

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_random_forest_regressor_sparse(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from onnx_extended.validation.cpu._validation import dense_to_sparse_struct
        from skl2onnx import to_onnx

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(3, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        expected = rf.predict(X[80:]).astype(numpy.float32).reshape((-1, 1))
        onx = to_onnx(rf, X[:1])
        feeds = {"X": X[80:]}

        # check with onnxruntime
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # transformation
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings)).replace(
            "BRANCH_", ""
        )
        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleRegressor",
            op_domain="ai.onnx.ml",
            new_op_type="TreeEnsembleRegressorSparse",
            new_op_domain="onnx_extented.ortops.optim.cpu",
            nodes_modes=modes,
        )
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))
        self.assertIn("TreeEnsembleRegressorSparse", str(onx2))

        # check with onnxruntime + custom op
        feeds = {"X": dense_to_sparse_struct(X[80:])}
        r = get_ort_ext_libs()
        self.assertExists(r[0])
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.ERROR)

    unittest.main(verbosity=2)
