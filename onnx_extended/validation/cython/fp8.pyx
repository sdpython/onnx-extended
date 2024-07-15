import numpy
cimport numpy as cnumpy
cimport cython
from libcpp cimport bool
from cython.cimports.libc.stdint import uint8_t, int64_t

# numpy.import_array()


cdef extern from "cpu/cast_fp8.h":
    void float_to_e4m3fn(int64_t n, const float* src, uint8_t* dst, bool saturate) nogil
    void e4m3fn_to_float(int64_t n, const uint8_t* src, float* dst) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cast_float32_to_e4m3fn(m, bool saturate = True):
    """
    Converts an array from float to float 8 e4m3fn.

    :param m: any array
    :param saturate: saturate the conversion
    :return: casted array
    """
    cdef cnumpy.ndarray cm = numpy.ascontiguousarray(m)
    cdef cnumpy.ndarray res = numpy.empty(m.shape, dtype=numpy.uint8)
    cdef int64_t n = m.size
    cdef const float* src = <float*> cm.data
    cdef uint8_t* dst = <uint8_t*> res.data
    with nogil:
        float_to_e4m3fn(n, src, dst, saturate)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cast_e4m3fn_to_float32(m):
    """
    Converts an array from float 8 e4m3fn to float.

    :param m: any array
    :return: casted array
    """
    cdef cnumpy.ndarray cm = numpy.ascontiguousarray(m)
    cdef cnumpy.ndarray res = numpy.empty(m.shape, dtype=numpy.float32)
    cdef int64_t n = m.size
    cdef const uint8_t* src = <uint8_t*> cm.data
    cdef float* dst = <float*> res.data
    with nogil:
        e4m3fn_to_float(n, src, dst)
    return res
