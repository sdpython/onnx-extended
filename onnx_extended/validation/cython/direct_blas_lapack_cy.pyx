import numpy
cimport numpy
cimport cython
numpy.import_array()
# cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_dgemm_dot(double* pa, double* pb,
                      int M, int N, int K,
                      int transA, int transB,
                      double* pc) noexcept nogil:
    """
    Wrapper for gemm.
    """

    cdef:
        char * cst = "nt"
        int lda = K if transA else M
        int ldb = K if transB else N
        int ldc = M
        double alpha = 1.
        double beta = 0.

    # cdef void dgemm(char *transa, char *transb,
    #                 int *m, int *n, int *k,
    #                 d *alpha,
    #                 d *a, int *lda,
    #                 d *b, int *ldb,
    #                 d *beta,
    #                 d *c, int *ldc)
    #                 noexcept nogil:

    cython_blas.dgemm(&cst[transA], &cst[transB],
                      &M, &N, &K,
                      &alpha, pa, &lda, pb, &ldb,
                      &beta, pc, &ldc)


@cython.boundscheck(False)
@cython.wraparound(False)
def dgemm_dot(double [:, ::1] A, double [:, ::1] B, transA, transB, double [:, ::1] C):
    """
    Wrapper for gemm.
    """

    cdef:
        int ta = 1 if transA else 0
        int tb = 1 if transB else 0
        int M = A.shape[1 - ta]
        int N = B.shape[tb]
        int K = A.shape[ta]

    c_dgemm_dot(&A[0, 0], &B[0, 0], M, N, K, ta, tb, &C[0, 0])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_sgemm_dot(float* pa, float* pb,
                      int M, int N, int K,
                      int transA, int transB,
                      float* pc) noexcept nogil:
    """
    Wrapper for gemm.
    """

    cdef:
        char * cst = "nt"
        int lda = K if transA else M
        int ldb = K if transB else N
        int ldc = M
        float alpha = 1.
        float beta = 0.

    cython_blas.sgemm(&cst[transA], &cst[transB],
                      &M, &N, &K,
                      &alpha, pa, &lda, pb, &ldb,
                      &beta, pc, &ldc)


@cython.boundscheck(False)
@cython.wraparound(False)
def sgemm_dot(float [:, ::1] A, float [:, ::1] B, transA, transB, float [:, ::1] C):
    """
    Wrapper for gemm.
    """

    cdef:
        int ta = 1 if transA else 0
        int tb = 1 if transB else 0
        int M = A.shape[1 - ta]
        int N = B.shape[tb]
        int K = A.shape[ta]

    c_sgemm_dot(&A[0, 0], &B[0, 0], M, N, K, ta, tb, &C[0, 0])
