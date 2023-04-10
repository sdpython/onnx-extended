import numpy
cimport numpy
cimport cython
numpy.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_sum_cy(float[:, :] m):
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


cdef extern from "vector_function.h" namespace "validation":
    float vector_sum(int nl, int nc, const float* values, int by_rows);


cdef float _vector_sum_c(float[:, :] m, int by_rows):
    return vector_sum(m.shape[0], m.shape[1], &(m[0,0]), by_rows);


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_sum_c(float[:, ::1] m, by_rows):
    """
    Computes the sum of all coefficients in a 2D tensor.

    :param m: 2D tensor
    :param by_rows: by rows or by columns
    :return: sum
    """
    return _vector_sum_c(m, by_rows);


cdef _vector_add_c(float[:, :] v1, float[:, :] v2, float[:, :] add):
    # C implementation but it needs another function
    # declared with def and not cdef.
    for i in range(0, v1.shape[0]):
        for j in range(0, v1.shape[1]):
            add[i, j] = v1[i, j] + v2[i, j]


def vector_add_c(v1, v2):
    """
    Computes the addition of two tensors of the same shape.

    :param v1: first tensor
    :param v2: second tensor
    :return: result.
    """
    if len(v1.shape) != 2:
        raise ValueError(
            f"This function only works for 2D tensors but shape is {v1.shape}.")
    if v1.shape != v2.shape:
        raise ValueError(f"Shapes must be equal but {v1.shape} != {v2.shape}.")
    if v1.dtype != v2.dtype:
        raise ValueError(f"Types must be equal but {v1.dtype} != {v2.dtype}.")
    out = numpy.array(v1.shape, dtype=v1.dtype)
    _vector_add_c(v1, v2, out)
    return out
