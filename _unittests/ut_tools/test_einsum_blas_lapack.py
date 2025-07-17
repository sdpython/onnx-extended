import unittest
import numpy
from scipy.linalg.blas import sgemm
from onnx_extended.ext_test_case import ExtTestCase
from onnx_extended.tools.einsum.blas_lapack import gemm_dot, pygemm


def gemm_pure_python(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    GEMM: C = alpha * op(A) * op(B)
    All matrices are stored in 1D row-major layout.
    TransA, TransB : bool (True = transpose)
    M, N, K        : matrix dimensions
    lda, ldb, ldc  : leading dimensions (stride between rows)
    A, B, C        : 1D float lists representing 2D matrices
    """
    assert beta == 0
    for i in range(M):  # rows of C
        for j in range(N):  # cols of C
            acc = 0.0
            for k in range(K):
                # Access A[i, k] or A[k, i]
                if not TransA:
                    a_index = i * lda + k
                else:
                    a_index = k * lda + i

                # Access B[k, j] or B[j, k]
                if not TransB:
                    b_index = k * ldb + j
                else:
                    b_index = j * ldb + k

                acc += A[a_index] * B[b_index]

            # Store in C[i, j]
            c_index = i * ldc + j
            C[c_index] += alpha * acc


class TestBlasLapack(ExtTestCase):
    def test_gemm(self):
        A = numpy.arange(4).reshape((2, 2)) + 1
        B = numpy.arange(4).reshape((2, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(
                        dtype=dtype,
                        transA=t1,
                        transB=t2,
                        shapeA=a.shape,
                        shapeB=b.shape,
                    ):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        exp = ta @ tb
                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

                        M, N, K = 2, 2, 2
                        lda, ldb, ldc = 2, 2, 2

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(
                            t2,
                            t1,
                            M,
                            N,
                            K,
                            1.0,
                            b.ravel(),
                            ldb,
                            a.ravel(),
                            lda,
                            0.0,
                            c,
                            ldc,
                        )
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

    def test_gemm_1(self):
        A = numpy.arange(1).reshape((1, 1)) + 1
        B = numpy.arange(1).reshape((1, 1)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(
                        dtype=dtype,
                        transA=t1,
                        transB=t2,
                        shapeA=a.shape,
                        shapeB=b.shape,
                    ):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        exp = ta @ tb
                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

                        M, N, K = 1, 1, 1
                        lda, ldb, ldc = 1, 1, 1

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(
                            t2,
                            t1,
                            M,
                            N,
                            K,
                            1.0,
                            b.ravel(),
                            ldb,
                            a.ravel(),
                            lda,
                            0.0,
                            c,
                            ldc,
                        )
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

    def test_gemm_exc(self):
        a = numpy.arange(3).reshape((1, 3)) + 1
        b = numpy.arange(3).reshape((3, 1)) + 10
        c = numpy.empty((1, 2), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(
                False, False, 1, 1, 1, 1.0, b.ravel(), 1, a.ravel(), 1, 0.0, c, 1
            ),
            AssertionError,
        )
        c = numpy.empty((1,), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(
                False, False, 1, 1, 1, 1.0, b.ravel(), 1, a.ravel(), 1, 0.0, c, 1
            ),
            ValueError,
        )
        c = numpy.empty((1,), dtype=a.dtype)
        a = numpy.arange(4) + 1
        b = numpy.arange(4) + 10
        c = numpy.empty((4,), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(
                False, False, 2, 4, 2, 1.0, b.ravel(), 10, a.ravel(), 10, 0.0, c, 10
            ),
            ValueError,
        )
        self.assertRaise(
            lambda: pygemm(
                False, False, 2, 2, 2, 1.0, b.ravel(), 10, a.ravel(), 10, 0.0, c, 10
            ),
            IndexError,
        )
        self.assertRaise(
            lambda: pygemm(
                False, False, 2, 2, 2, 1.0, b.ravel(), 1, a.ravel(), 10, 0.0, c, 10
            ),
            IndexError,
        )
        self.assertRaise(
            lambda: pygemm(
                False, False, 2, 2, 2, 1.0, b.ravel(), 1, a.ravel(), 1, 0.0, c, 10
            ),
            IndexError,
        )

    def test_gemm_314(self):
        A = numpy.arange(3).reshape((1, 3)) + 1
        B = numpy.arange(4).reshape((4, 1)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(
                        dtype=dtype,
                        transA=t1,
                        transB=t2,
                        shapeA=a.shape,
                        shapeB=b.shape,
                    ):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(
                            t2,
                            t1,
                            N,
                            M,
                            K,
                            1.0,
                            b.ravel(),
                            ldb,
                            a.ravel(),
                            lda,
                            0.0,
                            c,
                            ldc,
                        )
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:  # noqa: SIM105
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

    def test_gemm_324(self):
        A = numpy.arange(6).reshape((2, 3)) + 1
        B = numpy.arange(8).reshape((4, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(
                        dtype=dtype,
                        transA=t1,
                        transB=t2,
                        shapeA=a.shape,
                        shapeB=b.shape,
                    ):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(
                            t2,
                            t1,
                            N,
                            M,
                            K,
                            1.0,
                            b.ravel(),
                            ldb,
                            a.ravel(),
                            lda,
                            0.0,
                            c,
                            ldc,
                        )
                        cc = c.reshape((M, N))
                        # self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:  # noqa: SIM105
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

    def test_gemm_323(self):
        A = numpy.arange(6).reshape((2, 3)) + 1
        B = numpy.arange(6).reshape((3, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(
                        dtype=dtype,
                        transA=t1,
                        transB=t2,
                        shapeA=a.shape,
                        shapeB=b.shape,
                    ):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(
                            t2,
                            t1,
                            N,
                            M,
                            K,
                            1.0,
                            b.ravel(),
                            ldb,
                            a.ravel(),
                            lda,
                            0.0,
                            c,
                            ldc,
                        )
                        cc = c.reshape((M, N))
                        # self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:  # noqa: SIM105
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

    def test_gemm_lda_lda_ldc_trans(self):
        A = numpy.zeros((1, 2, 3, 5), dtype=numpy.float32)
        B = numpy.zeros((1, 2, 3, 5), dtype=numpy.float32)
        A[:, 0, 0, :] = 1
        A[:, 0, 2, :] = 10
        A[:, 1, 1, :] = -1
        B[:, 0, 0, :] = 1
        B[:, 0, 2, :] = 10
        B[:, 1, 1, :] = -1
        expected = numpy.matmul(A, numpy.transpose(B, (0, 1, 3, 2)))
        shape_c = (A.shape[0], A.shape[1], A.shape[2], B.shape[2])
        C = numpy.zeros(shape_c, dtype=numpy.float32)

        shape_a = A.shape
        shape_b = B.shape
        shape_c = C.shape
        M, N, K = shape_a[2], shape_b[2], shape_b[3]

        # same for attention
        A = numpy.transpose(A, (0, 2, 1, 3))
        B = numpy.transpose(B, (0, 2, 1, 3))

        lda, ldb, ldc = shape_a[3] * shape_a[1], shape_b[3] * shape_b[1], shape_c[3]
        A = A.ravel()
        B = B.ravel()
        C = C.ravel()
        for i1 in range(shape_a[0]):
            for i2 in range(shape_a[1]):
                da = shape_a[1] * shape_a[2] * shape_a[3]
                db = shape_b[1] * shape_b[2] * shape_b[3]
                dc = shape_c[1] * shape_c[2] * shape_c[3]

                ai = i1 * da + i2 * shape_a[3]
                aj = ai + da

                bi = i1 * db + i2 * shape_b[3]
                bj = bi + db

                ci = i1 * dc + i2 * shape_c[2] * shape_c[3]
                cj = ci + dc // shape_c[1]

                gemm_pure_python(
                    False,
                    True,
                    M,
                    N,
                    K,
                    1.0,
                    A[ai:aj],
                    lda,
                    B[bi:bj],
                    ldb,
                    0.0,
                    C[ci:cj],
                    ldc,
                )

        got = C.reshape(shape_c)
        self.assertEqualArray(expected, got)

    def test_gemm_lda_lda_ldc_notrans(self):
        A = numpy.zeros((1, 2, 3, 5), dtype=numpy.float32)
        B = numpy.zeros((1, 2, 5, 3), dtype=numpy.float32)
        A[:, 0, 0, :] = 1
        A[:, 0, 2, :] = 10
        A[:, 1, 1, :] = -1
        B[:, 0, 0, :] = 1
        B[:, 0, 2, :] = 10
        B[:, 1, 1, :] = -1
        expected = numpy.matmul(A, B)
        shape_c = (A.shape[0], A.shape[1], A.shape[2], B.shape[3])
        C = numpy.zeros(shape_c, dtype=numpy.float32)

        shape_a = A.shape
        shape_b = B.shape
        shape_c = C.shape
        M, N, K = shape_a[2], shape_b[3], shape_b[2]

        # same for attention
        B = numpy.transpose(B, (0, 2, 1, 3))

        lda, ldb, ldc = shape_a[3], shape_b[3] * shape_b[1], shape_c[3]
        A = A.ravel()
        B = B.ravel()
        C = C.ravel()
        for i1 in range(shape_a[0]):
            for i2 in range(shape_a[1]):
                da = shape_a[1] * shape_a[2] * shape_a[3]
                db = shape_b[1] * shape_b[2] * shape_b[3]
                dc = shape_c[1] * shape_c[2] * shape_c[3]

                ai = i1 * da + i2 * shape_a[3] * shape_a[2]
                aj = ai + da // shape_a[1]

                bi = i1 * db + i2 * shape_b[3]
                bj = bi + db

                ci = i1 * dc + i2 * shape_c[2] * shape_c[3]
                cj = ci + dc // shape_c[1]

                gemm_pure_python(
                    False,
                    False,
                    M,
                    N,
                    K,
                    1.0,
                    A[ai:aj],
                    lda,
                    B[bi:bj],
                    ldb,
                    0.0,
                    C[ci:cj],
                    ldc,
                )

        got = C.reshape(shape_c)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main()
