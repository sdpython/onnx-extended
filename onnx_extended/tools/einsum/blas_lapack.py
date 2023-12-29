def pygemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Pure python implementation of GEMM.
    """
    assert len(A.shape) == 1, "A must be a vector."
    assert len(B.shape) == 1, "B must be a vector."
    assert len(C.shape) == 1, "C must be a vector."
    if A.shape[0] != M * K:
        raise ValueError(
            f"Dimension mismatch for A.shape={A.shape!r} M={M!r} N={N!r} K={K!r}."
        )
    if B.shape[0] != N * K:
        raise ValueError(
            f"Dimension mismatch for B.shape={B.shape!r} M={M!r} N={N!r} K={K!r}."
        )
    if C.shape[0] != N * M:
        raise ValueError(
            f"Dimension mismatch for C.shape={C.shape!r} M={M!r} N={N!r} K={K!r}."
        )

    if transA:
        a_i_stride = lda
        a_k_stride = 1
    else:
        a_i_stride = 1
        a_k_stride = lda

    if transB:
        b_j_stride = 1
        b_k_stride = ldb
    else:
        b_j_stride = ldb
        b_k_stride = 1

    c_i_stride = 1
    c_j_stride = ldc

    n_loop = 0
    for j in range(N):
        for i in range(M):
            total = 0
            for k in range(K):
                n_loop += 1
                a_index = i * a_i_stride + k * a_k_stride
                if a_index >= A.shape[0]:
                    raise IndexError(
                        "A: i=%d a_index=%d >= %d "
                        "(a_i_stride=%d a_k_stride=%d)"
                        % (i, a_index, A.shape[0], a_i_stride, a_k_stride)
                    )
                a_val = A[a_index]

                b_index = j * b_j_stride + k * b_k_stride
                if b_index >= B.shape[0]:
                    raise IndexError(
                        "B: j=%d b_index=%d >= %d "
                        "(a_i_stride=%d a_k_stride=%d)"
                        % (j, b_index, B.shape[0], b_j_stride, b_k_stride)
                    )
                b_val = B[b_index]

                mult = a_val * b_val
                total += mult

            c_index = i * c_i_stride + j * c_j_stride
            if c_index >= C.shape[0]:
                raise IndexError("C: %d >= %d" % (c_index, C.shape[0]))
            C[c_index] = alpha * total + beta * C[c_index]

    assert n_loop == M * N * K, (
        "Unexpected number of loops: %d != %d = (%d * %d * %d) "
        "lda=%d ldb=%d ldc=%d" % (n_loop, M * N * K, M, N, K, lda, ldb, ldc)
    )


def gemm_dot(A, B, transA=False, transB=False):
    """
    Implements dot product, the implementation could be optimized
    with `scipy.linalg.blas.sgemm`.

    :param A: first matrix
    :param B: second matrix
    :param transA: is first matrix transposed?
    :param transB: is second matrix transposed?
    """
    assert (
        A.dtype == B.dtype
    ), f"Matrices A and B must have the same dtype not {A.dtype!r}, {B.dtype!r}."
    assert len(A.shape) == 2, f"Matrix A does not have 2 dimensions but {len(A.shape)}."
    assert len(B.shape) == 2, f"Matrix B does not have 2 dimensions but {len(B.shape)}."

    if transA:
        A = A.T
    if transB:
        B = B.T
    return A @ B
