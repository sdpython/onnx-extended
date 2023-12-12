import unittest
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor_value_info
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from onnx_extended.ortops.optim.cpu import documentation
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
)
from onnx_extended.ext_test_case import ExtTestCase, skipif_ci_apple

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

    @skipif_ci_apple("crash")
    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_random_forest_aregressor_sparse(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from onnx_extended.validation.cpu._validation import (
            dense_to_sparse_struct,
            sparse_struct_indices_values,
        )
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
        del onx2.graph.input[:]
        onx2.graph.input.append(make_tensor_value_info("X", TensorProto.FLOAT, (None,)))
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))
        self.assertIn("TreeEnsembleRegressorSparse", str(onx2))

        # check with onnxruntime + custom op
        sp = dense_to_sparse_struct(X[80:])
        indices, values = sparse_struct_indices_values(sp)
        self.assertEqualArray(numpy.arange(indices.size).astype(numpy.uint32), indices)
        feeds = {"X": sp}
        r = get_ort_ext_libs()
        self.assertExists(r[0])
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        indices2, values2 = sparse_struct_indices_values(sp)
        self.assertEqualArray(indices, indices2)
        self.assertEqualArray(values, values2)
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_apple("crash")
    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_random_forest_classifier_sparse(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from onnx_extended.validation.cpu._validation import dense_to_sparse_struct
        from skl2onnx import to_onnx

        X, y = make_classification(
            100,
            3,
            n_classes=3,
            n_informative=2,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=32,
        )
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        self.assertEqual(len(set(y)), 3)

        rf = RandomForestClassifier(500, max_depth=2, random_state=32)
        rf.fit(X[:80], y[:80])
        expected = rf.predict(X[80:]).astype(numpy.int64)
        expected_proba = rf.predict_proba(X[80:]).astype(numpy.float32)
        onx = to_onnx(rf, X[:1], options={"zipmap": False})
        feeds = {"X": X[80:]}

        # check with onnxruntime
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0])
        self.assertEqualArray(expected_proba, got[1], atol=1e-5)

        # transformation
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings)).replace(
            "BRANCH_", ""
        )
        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleClassifier",
            op_domain="ai.onnx.ml",
            new_op_type="TreeEnsembleClassifierSparse",
            new_op_domain="onnx_extented.ortops.optim.cpu",
            nodes_modes=modes,
        )
        del onx2.graph.input[:]
        onx2.graph.input.append(make_tensor_value_info("X", TensorProto.FLOAT, (None,)))
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))
        self.assertIn("TreeEnsembleClassifierSparse", str(onx2))

        # check with onnxruntime + custom op
        feeds = {"X": dense_to_sparse_struct(X[80:])}
        r = get_ort_ext_libs()
        self.assertExists(r[0])
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertEqualArray(expected_proba, got[1], atol=1e-5)
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.ERROR)

    unittest.main(verbosity=2)
