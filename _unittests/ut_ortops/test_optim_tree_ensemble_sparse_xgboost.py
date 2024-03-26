import unittest
import sys
import numpy
import scipy.sparse
from onnx import TensorProto
from onnx.helper import make_tensor_value_info
from sklearn.datasets import make_regression
import xgboost
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
)
from onnx_extended.ext_test_case import ExtTestCase, skipif_ci_apple, skipif_unstable

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None


class TestXGBoostSparse(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    @skipif_ci_apple("crash")
    @skipif_unstable("unstable on github workflow")
    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_xgbregressor_sparse(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from onnx_extended.validation.cpu._validation import dense_to_sparse_struct
        from skl2onnx import to_onnx, update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )

        # training with sparse
        X, y = make_regression(n_samples=400, n_features=10, random_state=0)
        mask = numpy.random.randint(0, 50, size=(X.shape)) != 0
        X[mask] = 0
        y = (y + mask.sum(axis=1, keepdims=0)).astype(numpy.float32)
        X_sp = scipy.sparse.coo_matrix(X)
        X = X.astype(numpy.float32)

        rf = xgboost.XGBRegressor(
            n_estimators=5, max_depth=4, random_state=0, base_score=0.5
        )
        rf.fit(X_sp, y)
        expected = rf.predict(X).astype(numpy.float32).reshape((-1, 1))
        expected_sparse = rf.predict(X_sp).astype(numpy.float32).reshape((-1, 1))
        if sys.platform != "win32":
            diff = numpy.abs(expected - expected_sparse)
            self.assertNotEqual(diff.min(), diff.max())

        # conversion to onnx
        update_registered_converter(
            xgboost.XGBRegressor,
            "XGBoostXGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
        )
        onx = to_onnx(rf, X[:1], target_opset={"ai.onnx.ml": 3})
        feeds = {"X": X}

        # check with onnxruntime
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        diff = expected - got  # xgboost converter has a bug
        self.assertEqualArray(diff.min(), diff.max(), atol=1e-4)
        diff = expected_sparse - got
        self.assertNotAlmostEqual(diff.min(), diff.max(), atol=1e-4)

        # with TreeEnsembleRegressorSparse
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
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))
        self.assertIn("TreeEnsembleRegressorSparse", str(onx2))

        onx2.graph.input.append(make_tensor_value_info("X", TensorProto.FLOAT, (None,)))
        for att in onx2.graph.node[0].attribute:
            if att.name == "nodes_missing_value_tracks_true":
                self.assertEqual({0, 1}, set(att.ints))

        # check with onnxruntime + custom op
        feeds = {"X": dense_to_sparse_struct(X)}
        r = get_ort_ext_libs()
        self.assertExists(r[0])
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        diff = expected - got  # xgboost converter has a bug
        self.assertNotAlmostEqual(diff.min(), diff.max(), 1e-4)
        diff = expected_sparse - got
        self.assertEqualArray(diff.min(), diff.max(), 1e-4)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.ERROR)

    unittest.main(verbosity=2)
