import numpy
cimport numpy
cimport cython
numpy.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_sum_cy(float[:, :, ::1] m):
    """
    Computes the sum of all coefficients in a matrix.

    :param m: 2D tensor
    :param by_rows: by rows or by columns
    :return: sum
    """
    cdef int i, j
    cdef int I = m.shape[0]
    cdef int J = m.shape[1]
    cdef float total = 0;
    with nogil:
        for i in range(0, I):
            for j in range(0, J):
                total += m[i, j]
    return total


cdef extern from "vector_sum.h":
    float vector_sum(int nl, int nc, const float* values, bool by_rows);

cdef float _vector_sum_c(float[::1, ::1] m, bool by_rows):
    return vector_sum(m.shape[0], m.shape[1], &(m[0,0]), by_rows);

@cython.boundscheck(False)
@cython.wraparound(False)
def vector_sum_c(float[::1, ::1] m, by_rows):
    """
    Computes the sum of all coefficients in a matrix.

    :param m: 2D tensor
    :param by_rows: by rows or by columns
    :return: sum
    """
    return _vector_sum_c(m, by_rows);
