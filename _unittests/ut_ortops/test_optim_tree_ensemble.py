import unittest
import numpy
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from onnx_extended.ortops.tutorial.cpu import documentation
from onnx_extended.ortops.optim.optimize import (
    change_onnx_operator_domain,
    get_node_attribute,
    optimize_model,
)
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.tools.onnx_nodes import convert_onnx_model
from onnx_extended.ext_test_case import ExtTestCase, skipif_ci_apple

try:
    from onnxruntime import InferenceSession, SessionOptions
except ImportError:
    SessionOptions, InferenceSession = None, None


class TestOrtOpOptimTreeEnsembleCpu(ExtTestCase):
    def test_get_ort_ext_libs(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        r = get_ort_ext_libs()
        self.assertEqual(len(r), 1)

    def test_documentation(self):
        doc = documentation()
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 4)
        for d in doc:
            self.assertIn("~~~~", d)
            self.assertIsInstance(d, str)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @skipif_ci_apple("crash")
    def test_random_forest_regressor(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
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

        # check with CReferenceEvaluator
        ref = CReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got.reshape((-1, 1)), atol=1e-5)

        # transformation
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings)).replace(
            "BRANCH_", ""
        )
        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleRegressor",
            op_domain="ai.onnx.ml",
            new_op_domain="onnx_extented.ortops.optim.cpu",
            nodes_modes=modes,
        )
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))

        # check with CReferenceEvaluator
        ref = CReferenceEvaluator(onx2)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got.reshape((-1, 1)), atol=1e-5)

        # check with onnxruntime + custom op
        r = get_ort_ext_libs()
        self.assertExists(r[0])
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    def test_tree_run_optimize_model(self):
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from skl2onnx import to_onnx

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(3, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        onx = to_onnx(rf, X[:1])

        optim_params = dict(
            parallel_tree=[20, 40],
            parallel_tree_N=[128],
            parallel_N=[50],
            batch_size_tree=[2],
            batch_size_rows=[2],
            use_node3=[0],
        )

        def transform_model(onx, **kwargs):
            att = get_node_attribute(onx.graph.node[0], "nodes_modes")
            modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))
            return change_onnx_operator_domain(
                onx,
                op_type="TreeEnsembleRegressor",
                op_domain="ai.onnx.ml",
                new_op_domain="onnx_extented.ortops.optim.cpu",
                nodes_modes=modes,
                **kwargs,
            )

        def create_session(onx):
            opts = SessionOptions()
            r = get_ort_ext_libs()
            if r is None:
                raise AssertionError("No custom implementation available.")
            opts.register_custom_ops_library(r[0])
            return InferenceSession(
                onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
            )

        batch_size = 40
        res = optimize_model(
            onx,
            feeds={"X": X[-batch_size:]},
            transform=transform_model,
            session=create_session,
            baseline=lambda onx: InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            ),
            params=optim_params,
            verbose=False,
            number=2,
            repeat=2,
            warmup=1,
        )
        self.assertEqual(len(res), 6)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @skipif_ci_apple("unstable")
    def test_random_forest_regressor_1000(self):
        """
        If is compiled with macro DEBUG_STEP enabled,
        the following strings should appear to make sure all
        paths are checked with this tests.

        ::

            "S:N1:TN"
            "S:N1:TN-P",
            "S:NN:TN",
            "S:NNB:TN-PG",
            "S:NN-P:TN",
        """
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
        from skl2onnx import to_onnx

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(500, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        expected = rf.predict(X[80:]).astype(numpy.float32).reshape((-1, 1))
        onx = to_onnx(rf, X[:1])

        # transformation
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))

        for params in [
            dict(
                parallel_tree=1000,
                parallel_tree_N=1000,
                parallel_N=1000,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
            dict(
                parallel_tree=1000,
                parallel_tree_N=40,
                parallel_N=10,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
            dict(
                parallel_tree=40,
                parallel_tree_N=1000,
                parallel_N=10,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
        ]:
            onx2 = change_onnx_operator_domain(
                onx,
                op_type="TreeEnsembleRegressor",
                op_domain="ai.onnx.ml",
                new_op_domain="onnx_extented.ortops.optim.cpu",
                nodes_modes=modes,
                **params,
            )
            self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))

            # check with onnxruntime + custom op
            r = get_ort_ext_libs()
            self.assertExists(r[0])
            opts = SessionOptions()
            opts.register_custom_ops_library(r[0])
            sess = InferenceSession(
                onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
            )

            feeds = {"X": X[80:81]}
            got = sess.run(None, feeds)[0]
            self.assertEqualArray(expected[:1], got, atol=1e-4)

            feeds = {"X": X[80:]}
            got = sess.run(None, feeds)[0]
            self.assertEqualArray(expected, got, atol=1e-4)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @skipif_ci_apple("crash")
    def test_random_forest_classifier_1000_multi(self):
        """
        If is compiled with macro DEBUG_STEP enabled,
        the following strings should appear to make sure all
        paths are checked with this tests.

        ::

            "M:N1:TN",
            "M:N1:TN-P",
            "M:NN:TN-P",
            "M:NNB:TN-PG",
            "M:NNB-P:TN",
        """
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
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
        expected_labels = rf.predict(X[80:]).astype(numpy.int64)
        expected = rf.predict_proba(X[80:]).astype(numpy.float32)
        onx = to_onnx(rf, X[:1], options={"zipmap": False})

        # transformation
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))

        for params in [
            dict(
                parallel_tree=1000,
                parallel_tree_N=1000,
                parallel_N=1000,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
            dict(
                parallel_tree=1000,
                parallel_tree_N=40,
                parallel_N=10,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
            dict(
                parallel_tree=40,
                parallel_tree_N=1000,
                parallel_N=10,
                batch_size_tree=2,
                batch_size_rows=2,
                use_node3=0,
            ),
        ]:
            onx2 = change_onnx_operator_domain(
                onx,
                op_type="TreeEnsembleClassifier",
                op_domain="ai.onnx.ml",
                new_op_domain="onnx_extented.ortops.optim.cpu",
                nodes_modes=modes,
                **params,
            )
            self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))

            # check with onnxruntime + custom op
            r = get_ort_ext_libs()
            self.assertExists(r[0])
            opts = SessionOptions()
            opts.register_custom_ops_library(r[0])
            sess = InferenceSession(
                onx2.SerializeToString(), opts, providers=["CPUExecutionProvider"]
            )

            feeds = {"X": X[80:81]}
            got = sess.run(None, feeds)
            self.assertEqualArray(expected[:1], got[1], atol=1e-4)
            self.assertEqualArray(expected_labels[:1], got[0], atol=1e-4)

            feeds = {"X": X[80:]}
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[1], atol=1e-4)
            self.assertEqualArray(expected_labels, got[0], atol=1e-4)

    @unittest.skipIf(InferenceSession is None, "onnxruntime not installed")
    @skipif_ci_apple("unstable")
    def test_random_forest_regressor_as_tensor(self):
        from skl2onnx import to_onnx
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        X, y = make_regression(100, 2, n_informative=1, random_state=32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        rf = RandomForestRegressor(3, max_depth=2, random_state=32, n_jobs=-1)
        rf.fit(X[:80], y[:80])
        expected = rf.predict(X[80:]).astype(numpy.float32).reshape((-1, 1))
        onx = to_onnx(rf, X[:1], target_opset={"ai.onnx.ml": 3, "": 18})
        onx = convert_onnx_model(
            onx,
            opsets={"": 18, "ai.onnx.ml": 3},
            use_as_tensor_attributes=True,
            verbose=0,
        )
        if "as_tensor" not in str(onx):
            try:
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
            except ImportError:
                onnx_simple_text_plot = str
            raise AssertionError(
                f"Unable to find as_tensor in\n{onnx_simple_text_plot(onx)}."
            )

        feeds = {"X": X[80:]}

        # check with CReferenceEvaluator
        ref = CReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got.reshape((-1, 1)), atol=1e-5)

        # check with onnxruntime
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # transformation
        opsets = {d.domain: d.version for d in onx.opset_import}
        self.assertEqual({"ai.onnx.ml": 3, "": 18}, opsets)
        att = get_node_attribute(onx.graph.node[0], "nodes_modes")
        modes = ",".join(map(lambda s: s.decode("ascii"), att.strings))
        onx2 = change_onnx_operator_domain(
            onx,
            op_type="TreeEnsembleRegressor",
            op_domain="ai.onnx.ml",
            new_op_domain="onnx_extented.ortops.optim.cpu",
            nodes_modes=modes,
            new_opset=3,
        )
        self.assertIn("onnx_extented.ortops.optim.cpu", str(onx2))
        opsets = {d.domain: d.version for d in onx2.opset_import}
        self.assertEqual(
            {"ai.onnx.ml": 3, "": 18, "onnx_extented.ortops.optim.cpu": 3}, opsets
        )

        # check with CReferenceEvaluator
        ref = CReferenceEvaluator(onx2)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got.reshape((-1, 1)), atol=1e-5)

        # check with onnxruntime + custom op
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
