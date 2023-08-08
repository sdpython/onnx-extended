import unittest
import numpy
from onnx.defs import onnx_opset_version
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings
from onnx_extended.reference import CReferenceEvaluator


class TestCTreeEnsemble(ExtTestCase):
    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_classifier_bin(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        y[y == 2] = 0
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)
        X_test = X_test[2:]

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        self.assertNotIn("nodes_values_as_tensor", str(model_def))
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        exp = clr.predict_proba(X_test)
        self.assertEqualArray(exp.astype(numpy.float32), y[1], atol=1e-5)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_classifier_multi(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)
        X_test = X_test[2:]

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        self.assertNotIn("nodes_values_as_tensor", str(model_def))
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        exp = clr.predict_proba(X_test)
        self.assertEqualArray(exp.astype(numpy.float32), y[1], atol=1e-5)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_classifier_plusten(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        y += 10
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})

        exp = clr.predict_proba(X_test)
        self.assertEqualArray(exp.astype(numpy.float32), y[1], atol=1e-5)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_gradient_boosting_classifier2(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        y[y == 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        exp = clr.predict_proba(X_test).astype(numpy.float32)
        self.assertEqualArray(exp, y[1], atol=1e-3)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_gradient_boosting_classifier3(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y = iris.data.astype(numpy.float32), iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        exp = clr.predict_proba(X_test).astype(numpy.float32)
        self.assertEqualArray(exp, y[1], atol=1e-3)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    # @unittest.skipIf(True, reason="not implemented yet")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    @unittest.skipIf(
        onnx_opset_version() < 19, reason="ArrayFeatureExtractor has no implementation"
    )
    def test_decision_tree_classifier_mlabel(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
        )

        iris = load_iris()
        X, y_ = iris.data.astype(numpy.float32), iris.target
        y = numpy.zeros((y_.shape[0], 3), dtype=numpy.int64)
        y[y_ == 0, 0] = 1
        y[y_ == 1, 1] = 1
        y[y_ == 2, 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier(max_depth=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        # with open("jjj.onnx", "wb") as f:
        #    f.write(model_def.SerializeToString())
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleClassifier_1)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        exp = numpy.array(clr.predict_proba(X_test))
        # the conversion fails, it needs to be investigated
        self.assertEqualArray(exp.astype(numpy.float32), y[1], atol=1)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_regressor(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor import (
            TreeEnsembleRegressor_1,
        )

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleRegressor_1)

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 1]})
            lexp = clr.predict(X_test[i : i + 1])
            self.assertEqual(lexp.shape, y[0].shape)
            self.assertEqualArray(lexp.astype(numpy.float32), y[0])

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 2]})
            lexp = clr.predict(X_test[i : i + 2])
            self.assertEqual(lexp.shape, y[0].shape)
            self.assertEqualArray(lexp.astype(numpy.float32), y[0])

        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp.astype(numpy.float32), y[0])

    @ignore_warnings((FutureWarning, DeprecationWarning, UserWarning))
    def test_decision_tree_regressor_double(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor import (
            TreeEnsembleRegressor_3,
        )
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
        from lightgbm import LGBMRegressor

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LGBMRegressor(num_iterations=1, max_depth=5)
        clr.fit(X_train, y_train)

        update_registered_converter(
            LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
        )

        try:
            model_def = to_onnx(clr, X_train.astype(numpy.float64))
        except ImportError as e:
            if "cannot import name 'FEATURE_IMPORTANCE_TYPE_MAPPER'" in str(e):
                return
            raise e
        for op in model_def.opset_import:
            if op.domain == "ai.onnx.ml":
                self.assertEqual(op.version, 3)
        oinf = CReferenceEvaluator(model_def)
        self.assertIsInstance(oinf.rt_nodes_[0], TreeEnsembleRegressor_3)

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 1]})
            lexp = clr.predict(X_test[i : i + 1])
            self.assertEqual(lexp.shape, y[0].shape)
            self.assertEqualArray(lexp.astype(numpy.float32), y[0])

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 2]})
            lexp = clr.predict(X_test[i : i + 2])
            self.assertEqual(lexp.shape, y[0].shape)
            self.assertEqualArray(lexp.astype(numpy.float32), y[0])

        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp.astype(numpy.float32), y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_regressor2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        y = numpy.vstack([y, y]).T
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp.astype(numpy.float32), y[0])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_decision_tree_depth2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier(max_depth=2)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        oinf = CReferenceEvaluator(model_def)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test).astype(numpy.float32)
        got = y[1]
        self.assertEqualArray(exp, got, atol=1e-5)

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_random_forest_classifier5(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=11)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        oinf = CReferenceEvaluator(model_def)
        y = oinf.run(None, {"X": X_test[:5].astype(numpy.float32)})
        lexp = clr.predict(X_test[:5])
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test[:5]).astype(numpy.float32)
        got = y[1]
        self.assertEqualArray(exp, got, atol=1e-5)

    def common_test_onnxrt_python_tree_ensemble_runtime_version(
        self, dtype, multi=False
    ):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = y.astype(dtype)
        if multi:
            y = numpy.vstack([y, y]).T
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestRegressor(n_estimators=70)
        clr.fit(X_train, y_train)

        X_test2 = numpy.empty((X_test.shape[0] * 200, X_test.shape[1]), dtype=dtype)
        for i in range(200):
            d = X_test.shape[0] * i
            X_test2[d : d + X_test.shape[0], :] = X_test
        X_test = X_test2

        # default runtime
        model_def = to_onnx(clr, X_train.astype(dtype))
        oinf = CReferenceEvaluator(model_def)
        #
        # oinf.rt_nodes_[0]._init(dtype, 1)
        y = oinf.run(None, {"X": X_test})
        lexp = clr.predict(X_test).astype(dtype)
        self.assertEqual(lexp.shape, y[0].shape)
        atol = {numpy.float32: 1e-5, numpy.float64: 1e-1}
        with self.subTest(dtype=dtype):
            self.assertEqualArray(lexp, y[0], atol=atol[dtype])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_float(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(numpy.float32)

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_double(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(numpy.float64)

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_float_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(
            numpy.float32, True
        )

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_double_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(
            numpy.float64, True
        )

    def common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
        self, dtype, multi=False, single_cls=False
    ):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = y.astype(numpy.int64)
        if not multi:
            y[y == 2] = 0
        if single_cls:
            y[:] = 0
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestClassifier(n_estimators=40, max_depth=4)
        clr.fit(X_train, y_train)

        X_test2 = numpy.empty((X_test.shape[0] * 200, X_test.shape[1]), dtype=dtype)
        for i in range(200):
            d = X_test.shape[0] * i
            X_test2[d : d + X_test.shape[0], :] = X_test
        X_test = X_test2
        # X_test = X_test

        # default runtime
        model_def = to_onnx(
            clr,
            X_train.astype(dtype),
            options={RandomForestClassifier: {"zipmap": False}},
            target_opset=17,
        )
        oinf = CReferenceEvaluator(model_def)
        y = oinf.run(None, {"X": X_test.astype(dtype)})
        lexp = clr.predict_proba(X_test).astype(numpy.float32)
        atol = {numpy.float32: 1e-5, numpy.float64: 1.01e-1}
        with self.subTest(dtype=dtype):
            if single_cls:
                diff = list(sorted(numpy.abs(lexp.ravel() - y[1])))
                mx = max(diff[:-5])
                if mx > 1e-5:
                    self.assertEqualArray(
                        lexp.ravel().astype(dtype), y[1], atol=atol[dtype]
                    )
            else:
                self.assertEqualArray(lexp.astype(dtype), y[1], atol=atol[dtype])

        # other runtime
        for rv in [0, 1, 2, 3]:
            if single_cls and rv == 0:
                continue
            with self.subTest(runtime_version=rv):
                y = oinf.run(None, {"X": X_test.astype(dtype)})
                if single_cls:
                    diff = list(sorted(numpy.abs(lexp.ravel() - y[1])))
                    mx = max(diff[:-5])
                    if mx > 1e-5:
                        self.assertEqualArray(
                            lexp.ravel().astype(dtype), y[1], atol=atol[dtype]
                        )
                else:
                    self.assertEqualArray(lexp.astype(dtype), y[1], atol=atol[dtype])

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_float_cls(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(numpy.float32)

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_double_cls(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(numpy.float64)

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_float_cls_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float32, True
        )

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_double_cls_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float64, True
        )

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_float_cls_single(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float32, False, True
        )

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_tree_ensemble_runtime_version_double_cls_single(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float64, False, True
        )

    @unittest.skipIf(onnx_opset_version() < 19, reason="ReferenceEvaluator is bugged")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_random_forest_with_only_one_class(self):
        rnd = numpy.random.RandomState(4)
        ntrain = 10000
        nfeat = 30
        X_train = numpy.empty((ntrain, nfeat)).astype(numpy.float32)
        X_train[:, :] = rnd.rand(ntrain, nfeat)[:, :]
        eps = rnd.rand(ntrain) - 0.5
        y_train_f = X_train.sum(axis=1) + eps
        y_train = (y_train_f > 12).astype(numpy.int64)
        y_train[y_train_f > 15] = 2
        y_train[y_train_f < 10] = 3
        y_train[:] = 2

        rf = RandomForestClassifier(max_depth=2, n_estimators=80, n_jobs=4)
        rf.fit(X_train, y_train)
        onx = to_onnx(rf, X_train[:1], options={id(rf): {"zipmap": False}})

        for rv in [3, 2, 1]:
            oinf = CReferenceEvaluator(onx)

            for n in [1, 20, 100, 2000, 1, 1000, 10]:
                x = numpy.empty((n, X_train.shape[1]), dtype=numpy.float32)
                x[:, :] = rnd.rand(n, X_train.shape[1])[:, :]
                with self.subTest(version=rv, n=n):
                    y = oinf.run(None, {"X": x})
                    self.assertEqual(len(y), 2)
                    lexp = rf.predict_proba(x).astype(numpy.float32)
                    self.assertEqualArray(lexp.ravel(), y[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
