import numpy
cimport numpy
cimport cython
from cpython cimport Py_buffer
from cpython.buffer cimport (
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_SIMPLE,
)
numpy.import_array()


cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        string()
        string(const char*)
        string(const char*, size_t)
        string(const string&)
        string& operator=(const char*)
        string& operator=(const string&)
        char& operator[](size_t)
        const char& operator[](size_t)
        const char* c_str()


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        int size() const
        const char* c_str() const


cdef extern from "ortapi.h" namespace "ortapi":

    vector[string] get_available_providers()
    void* create_session()
    void delete_session(void*)
    size_t get_input_count(void*)
    size_t get_output_count(void*)
    void session_load_from_file(void*, const char* filename)
    void session_load_from_bytes(void*, const void* buffer, size_t size)

    # void* ort_run_inference(void* inst, int n_inputs,
    # const char** input_names, void* ort_values, int& n_outputs);

    # https://stackoverflow.com/questions/46510654/using-pycapsule-in-cython
    # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/squeezenet/main.cpp


cdef list _ort_get_available_providers():
    """
    Returns the list of available providers.
    """
    cdef vector[string] prov = get_available_providers()
    cdef str resultat
    res = []
    for i in range(0, prov.size()):
        resultat = prov[i].c_str().decode('UTF-8', 'strict')
        res.append(resultat)
    return res


def ort_get_available_providers():
    """
    Returns the list of available providers.
    """
    r = _ort_get_available_providers()
    return r


cdef class OrtSession:
    """
    Wrapper around `onnxruntime C API <https://onnxruntime.ai/docs/api/c/>`_.
    """

    cdef void* session

    cdef void _session_load_from_bytes(self, object data):
        cdef Py_buffer buffer
        cdef char * ptr

        PyObject_GetBuffer(data, &buffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        try:
            ptr = <char *>buffer.buf
            session_load_from_bytes(self.session, <void*>ptr, len(data))
        finally:
            PyBuffer_Release(&buffer)

    def __init__(self, filename):
        self.session = <void*>create_session()
        if isinstance(filename, str):
            session_load_from_file(self.session, filename.encode('utf-8'))
        elif isinstance(filename, bytes):
            self._session_load_from_bytes(filename)
        else:
            raise TypeError(f"Unexpected type for filename {type(filename)}.")

    def __dealloc__(self):
        delete_session(self.session)

    def get_input_count(self):
        return get_input_count(self.session)

    def get_output_count(self):
        return get_output_count(self.session)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def run_inference_float(OrtSession inst, numpy.ndarray value):
    """
    Runs the inference on one value only.
    """
    raise NotImplementedError()


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def run_inference(inst, input_names, values):
    """
    Runs inference and returns the results.
    """
    raise NotImplementedError()
