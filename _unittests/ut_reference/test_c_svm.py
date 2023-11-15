import unittest
from logging import getLogger
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, SVR, OneClassSVM
from sklearn.exceptions import ConvergenceWarning
from onnx_extended.ext_test_case import ExtTestCase, ignore_warnings
from onnx_extended.reference import CReferenceEvaluator


def _modify_dimension(X, n_features, seed=19):
    """
    Modifies the number of features to increase
    or reduce the number of features.

    :param X: features matrix
    :param n_features: number of features
    :param seed: random seed (to get the same dataset at each call)
    :return: new featurs matrix
    """
    if n_features is None or n_features == X.shape[1]:
        return X
    if n_features < X.shape[1]:
        return X[:, :n_features]
    rstate = numpy.random.RandomState(seed)  # pylint: disable=E1101
    res = numpy.empty((X.shape[0], n_features), dtype=X.dtype)
    res[:, : X.shape[1]] = X[:, :]
    div = max((n_features // X.shape[1]) + 1, 2)
    for i in range(X.shape[1], res.shape[1]):
        j = i % X.shape[1]
        col = X[:, j]
        if X.dtype in (numpy.float32, numpy.float64):
            sigma = numpy.var(col) ** 0.5
            rnd = rstate.randn(len(col)) * sigma / div
            col2 = col + rnd
            res[:, j] -= col2 / div
            res[:, i] = col2
        elif X.dtype in (numpy.int32, numpy.int64):
            perm = rstate.permutation(col)
            h = rstate.randint(0, div) % X.shape[0]
            col2 = col.copy()
            col2[h::div] = perm[h::div]  # pylint: disable=E1136
            res[:, i] = col2
            h = (h + 1) % X.shape[0]
            res[h, j] = perm[h]  # pylint: disable=E1136
        else:  # pragma: no cover
            raise NotImplementedError(  # pragma: no cover
                f"Unable to add noise to a feature for this type {X.dtype}"
            )
    return res


class TestCSVM(ExtTestCase):
    def setUp(self):
        logger = getLogger("skl2onnx")
        logger.disabled = True
        return self

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svr(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVR()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test).reshape((-1, 1)).astype(numpy.float32)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svr_double(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVR()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float64))
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float64)})
        lexp = clr.predict(X_test).reshape((-1, 1))
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svr_20(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X = _modify_dimension(X, 20)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVR()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        lexp = clr.predict(X_test).reshape((-1, 1)).astype(numpy.float32)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_one_class_svm(self):
        from skl2onnx import to_onnx

        X = numpy.array([[0, 1, 2], [44, 36, 18], [-4, -7, -5]], dtype=numpy.float32)

        for kernel in ["linear", "sigmoid", "rbf", "poly"]:
            with self.subTest(kernel=kernel):
                model = OneClassSVM(kernel=kernel).fit(X)
                X32 = X.astype(numpy.float32)
                model_onnx = to_onnx(model, X32)
                cref = CReferenceEvaluator(model_onnx)
                res = cref.run(None, {"X": X32})
                scores = res[1]
                dec = (
                    model.decision_function(X32).astype(numpy.float32).reshape((-1, 1))
                )
                self.assertEqualArray(dec, scores, atol=1e-4)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svc_proba(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVC(probability=True)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y[1].values
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)
        self.assertEqualArray(lprob, got, atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svc_proba_20(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X = _modify_dimension(X, 20)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVC(probability=True)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y[1].values
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)
        self.assertEqualArray(lprob, got, atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svc_proba_double_20(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X = _modify_dimension(X, 20)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVC(probability=True)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float64), options={"zipmap": False}
        )
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float64)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y[1].values
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)
        self.assertEqualArray(lprob, got, atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svc_proba_linear(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearSVC()
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        lprob = clr.decision_function(X_test)
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqual(lprob.shape, y[1].shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)
        self.assertEqualArray(lprob, y[1], atol=1e-5)

    @ignore_warnings([FutureWarning, UserWarning, ConvergenceWarning, RuntimeWarning])
    def test_python_svc_proba_bin(self):
        from skl2onnx import to_onnx

        iris = load_iris()
        X, y = iris.data, iris.target
        y[y == 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVC(probability=True)
        clr.fit(X_train, y_train)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32), options={"zipmap": False}
        )
        cref = CReferenceEvaluator(model_def)
        y = cref.run(None, {"X": X_test.astype(numpy.float32)})
        self.assertEqual(len(y), 2)
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y[1].values
        self.assertEqual(lexp.shape, y[0].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqualArray(lexp, y[0], atol=1e-5)
        self.assertEqualArray(lprob, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
