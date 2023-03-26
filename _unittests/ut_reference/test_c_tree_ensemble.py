import unittest

import numpy
import pandas

from sklearn.datasets import load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skl2onnx import to_onnx
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings
from onnx_extended.reference import CReferenceEvaluator


class TestCTreeEnsemble(ExtTestCase):
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTreeClassifier(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test)
        got = y[1]
        self.assertEqualArray(exp, got, decimal=5)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTreeClassifier_plusten(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y += 10
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test)
        got = y[1]
        self.assertEqualArray(exp, got, decimal=5)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_GradientBoostingClassifier2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y[y == 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test)
        got = y[1]
        self.assertEqualArray(exp, got, decimal=3)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_GradientBoostingClassifier3(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["output_label", "output_probability"])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test)
        got = y[1]
        self.assertEqualArray(exp, got, decimal=3)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTreeClassifier_mlabel(self):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y = numpy.zeros((y_.shape[0], 3), dtype=int)
        y[y_ == 0, 0] = 1
        y[y_ == 1, 1] = 1
        y[y_ == 2, 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["output_label", "output_probability"])
        exp = numpy.array(clr.predict_proba(X_test))
        exp = exp.reshape(max(exp.shape), -1)
        p = y["output_probability"]
        got = pandas.DataFrame(p.values, columns=p.columns)
        self.assertEqualArray(exp, got, decimal=5)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTreeRegressor(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 1]})
            self.assertEqual(list(sorted(y)), ["variable"])
            lexp = clr.predict(X_test[i : i + 1])
            self.assertEqual(lexp.shape, y["variable"].shape)
            self.assertEqualArray(lexp, y["variable"])

        for i in range(0, 20):
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)[i : i + 2]})
            self.assertEqual(list(sorted(y)), ["variable"])
            lexp = clr.predict(X_test[i : i + 2])
            self.assertEqual(lexp.shape, y["variable"].shape)
            self.assertEqualArray(lexp, y["variable"])

        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["variable"])
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y["variable"].shape)
        self.assertEqualArray(lexp, y["variable"])

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTreeRegressor2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = numpy.vstack([y, y]).T
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["variable"])
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y["variable"].shape)
        self.assertEqualArray(lexp, y["variable"])

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_DecisionTree_depth2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier(max_depth=2)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["output_label", "output_probability"])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test)
        got = y[1]
        self.assertEqualArray(exp, got, decimal=5)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_RandomForestClassifer5(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=11)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = CReferenceEvaluator(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run(None, {"X": X_test[:5].astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ["output_label", "output_probability"])
        lexp = clr.predict(X_test[:5])
        self.assertEqualArray(lexp, y[0])

        exp = clr.predict_proba(X_test[:5])
        got = y[1]
        self.assertEqualArray(exp, got, decimal=5)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_openmp_compilation_p(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            RuntimeTreeEnsembleRegressorPFloat,
        )

        ru = RuntimeTreeEnsembleRegressorPFloat(1, 1, 1, False, False)
        r = ru.runtime_options()
        self.assertEqual("OPENMP", r)
        nb = ru.omp_get_max_threads()
        self.assertGreater(nb, 0)

        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            RuntimeTreeEnsembleClassifierPFloat,
        )

        ru = RuntimeTreeEnsembleClassifierPFloat(1, 1, 1, False, False)
        r = ru.runtime_options()
        self.assertEqual("OPENMP", r)
        nb2 = ru.omp_get_max_threads()
        self.assertEqual(nb2, nb)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_openmp_compilation_p_true(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            RuntimeTreeEnsembleRegressorPFloat,
        )

        ru = RuntimeTreeEnsembleRegressorPFloat(1, 1, 1, True, False)
        r = ru.runtime_options()
        self.assertEqual("OPENMP", r)
        nb = ru.omp_get_max_threads()
        self.assertGreater(nb, 0)

        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            RuntimeTreeEnsembleClassifierPFloat,
        )

        ru = RuntimeTreeEnsembleClassifierPFloat(1, 1, 1, True, False)
        r = ru.runtime_options()
        self.assertEqual("OPENMP", r)
        nb2 = ru.omp_get_max_threads()
        self.assertEqual(nb2, nb)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_average(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_average,
        )

        confs = [
            [100, 128, 100, False, False, True],
            [100, 128, 100, False, False, False],
            [10, 128, 10, False, False, True],
            [10, 128, 10, False, False, False],
            [2, 128, 2, False, False, True],
            [2, 128, 2, False, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_average(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_average(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_average_true(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_average,
        )

        confs = [
            [100, 128, 100, True, False, True],
            [100, 128, 100, True, False, False],
            [10, 128, 10, True, False, True],
            [10, 128, 10, True, False, False],
            [2, 128, 2, True, False, True],
            [2, 128, 2, True, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_average(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_average(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_sum(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_sum,
        )

        confs = [
            [100, 128, 100, False, False, True],
            [100, 128, 100, False, False, False],
            [10, 128, 10, False, False, True],
            [10, 128, 10, False, False, False],
            [2, 128, 2, False, False, True],
            [2, 128, 2, False, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_sum(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_sum(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_sum_true(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_sum,
        )

        confs = [
            [100, 128, 100, True, False, True],
            [100, 128, 100, True, False, False],
            [10, 128, 10, True, False, True],
            [10, 128, 10, True, False, False],
            [2, 128, 2, True, False, True],
            [2, 128, 2, True, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_sum(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_sum(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_min(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_min,
        )

        confs = [
            [100, 128, 100, False, False, True],
            [100, 128, 100, False, False, False],
            [10, 128, 10, False, False, True],
            [10, 128, 10, False, False, False],
            [2, 128, 2, False, False, True],
            [2, 128, 2, False, False, False],
        ]
        for conf in reversed(confs):
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_min(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_min(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_min_true(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_min,
        )

        confs = [
            [100, 128, 100, True, False, True],
            [100, 128, 100, True, False, False],
            [10, 128, 10, True, False, True],
            [10, 128, 10, True, False, False],
            [2, 128, 2, True, False, True],
            [2, 128, 2, True, False, False],
        ]
        for conf in reversed(confs):
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_min(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_min(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_max(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_max,
        )

        confs = [
            [100, 128, 100, False, False, True],
            [100, 128, 100, False, False, False],
            [10, 128, 10, False, False, True],
            [10, 128, 10, False, False, False],
            [2, 128, 2, False, False, True],
            [2, 128, 2, False, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_max(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_max(*(conf + [b, True]))

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_cpp_max_true(self):
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_p_ import (
            test_tree_regressor_multitarget_max,
        )

        confs = [
            [100, 128, 100, True, False, True],
            [100, 128, 100, True, False, False],
            [10, 128, 10, True, False, True],
            [10, 128, 10, True, False, False],
            [2, 128, 2, True, False, True],
            [2, 128, 2, True, False, False],
        ]
        for conf in confs:
            with self.subTest(conf=tuple(conf)):
                for b in [False, True]:
                    test_tree_regressor_multitarget_max(*(conf + [b, False]))
                for b in [False, True]:
                    test_tree_regressor_multitarget_max(*(conf + [b, True]))

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
        oinf.sequence_[0].ops_._init(dtype, 1)
        y = oinf.run(None, {"X": X_test})
        self.assertEqual(list(sorted(y)), ["variable"])
        lexp = clr.predict(X_test).astype(dtype)
        self.assertEqual(lexp.shape, y["variable"].shape)
        decimal = {numpy.float32: 5, numpy.float64: 1}
        with self.subTest(dtype=dtype):
            self.assertEqualArray(lexp, y["variable"], decimal=decimal[dtype])

        # other runtime
        for rv in [0, 1, 2, 3]:
            with self.subTest(runtime_version=rv):
                oinf.sequence_[0].ops_._init(dtype, rv)
                y = oinf.run(None, {"X": X_test})
                self.assertEqualArray(lexp, y["variable"], decimal=decimal[dtype])
        with self.subTest(runtime_version=40):
            self.assertRaise(
                lambda: oinf.sequence_[0].ops_._init(dtype, 40),
                ValueError,
            )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_float(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(numpy.float32)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_double(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(numpy.float64)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_float_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version(
            numpy.float32, True
        )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_double_multi(self):
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
        clr = RandomForestClassifier(n_estimators=40)
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
        for op in oinf.sequence_:
            if hasattr(op.ops_, "_init"):
                op.ops_._init(dtype, 1)
        y = oinf.run(None, {"X": X_test.astype(dtype)})
        self.assertEqual(list(sorted(y)), ["label", "probabilities"])
        lexp = clr.predict_proba(X_test)
        decimal = {numpy.float32: 5, numpy.float64: 1}
        with self.subTest(dtype=dtype):
            if single_cls:
                diff = list(sorted(numpy.abs(lexp.ravel() - y["probabilities"])))
                mx = max(diff[:-5])
                if mx > 1e-5:
                    self.assertEqualArray(
                        lexp.ravel(), y["probabilities"], decimal=decimal[dtype]
                    )
            else:
                self.assertEqualArray(lexp, y["probabilities"], decimal=decimal[dtype])

        # other runtime
        for rv in [0, 1, 2, 3]:
            if single_cls and rv == 0:
                continue
            with self.subTest(runtime_version=rv):
                for op in oinf.sequence_:
                    if hasattr(op.ops_, "_init"):
                        op.ops_._init(dtype, rv)
                y = oinf.run(None, {"X": X_test.astype(dtype)})
                if single_cls:
                    diff = list(sorted(numpy.abs(lexp.ravel() - y["probabilities"])))
                    mx = max(diff[:-5])
                    if mx > 1e-5:
                        self.assertEqualArray(
                            lexp.ravel(), y["probabilities"], decimal=decimal[dtype]
                        )
                else:
                    self.assertEqualArray(
                        lexp, y["probabilities"], decimal=decimal[dtype]
                    )

        with self.subTest(runtime_version=40):
            self.assertRaise(
                lambda: oinf.sequence_[0].ops_._init(dtype, 40),
                ValueError,
            )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_float_cls(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(numpy.float32)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_double_cls(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(numpy.float64)

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_float_cls_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float32, True
        )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_double_cls_multi(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float64, True
        )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_float_cls_single(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float32, False, True
        )

    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_onnxrt_python_tree_ensemble_runtime_version_double_cls_single(self):
        self.common_test_onnxrt_python_tree_ensemble_runtime_version_cls(
            numpy.float64, False, True
        )

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
            oinf.sequence_[0].ops_._init(numpy.float32, rv)

            for n in [1, 20, 100, 10000, 1, 1000, 10]:
                x = numpy.empty((n, X_train.shape[1]), dtype=numpy.float32)
                x[:, :] = rnd.rand(n, X_train.shape[1])[:, :]
                with self.subTest(version=rv, n=n):
                    y = oinf.run(None, {"X": x})[1]
                    lexp = rf.predict_proba(x)
                    self.assertEqualArray(lexp.ravel(), y, decimal=5)


if __name__ == "__main__":
    TestCTreeEnsemble().test_openmp_compilation_p_true()
    unittest.main(verbosity=2)
