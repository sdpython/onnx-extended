import numpy
cimport numpy
cimport cython
from libc.stdint cimport int64_t
from libc.string cimport memcpy 
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

    cdef cppclass OrtShape:
        OrtShape()
        void init(size_t)
        size_t ndim()
        void set(size_t i, int64_t dim)
        void* dims()

    cdef cppclass OrtCpuValue:
        OrtCpuValue()
        void init(size_t size, int elem_type, void* data, void* ort_value)
        void free_ort_value()
        int elem_type()
        size_t size()
        void* data()

    vector[string] get_available_providers()

    void* create_session()
    void delete_session(void*)
    size_t session_get_input_count(void*)
    size_t session_get_output_count(void*)
    void session_load_from_file(void*, const char* filename)
    void session_load_from_bytes(void*, const void* buffer, size_t size)
    void session_initialize(void*,
                            const char* optimized_file_path,
                            int graph_optimization_level,
                            int enable_cuda,
                            int cuda_device_id,
                            int set_denormal_as_zero,
                            int intra_op_num_threads,
                            int inter_op_num_threads)
    size_t session_run(void*,
                       size_t n_inputs,
                       const OrtShape* shapes,
                       const OrtCpuValue* values,
                       size_t max_outputs,
                       OrtShape* out_shapes,
                       OrtCpuValue* out_values)

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
    Wrapper around :epkg:`onnxruntime C API` based on :epkg:`cython`.

    :param filename: filename (str) or a bytes for a model serialized in
        memory
    :param graph_optimisation_level: level of graph optimisation,
        nodes fusion, see :epkg:`onnxruntime Graph Optimizations`
    :param enable_cuda: use CUDA provider
    :param cuda_device_id: CUDA device id
    :param set_denormal_as_zero: if a tensor contains too many denormal numbers,
        the execution is slowing down
    :param optimized_file_path: to write the optimized model
    :param inter_op_num_threads: number of threads used to parallelize
        the execution of the graph
    :param intra_op_num_threads: number of threads used to parallelize
        the execution within nodes
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

    def __init__(
        self,
        filename,
        graph_optimization_level=-1,
        enable_cuda=False,
        cuda_device_id=0,
        set_denormal_as_zero=False,
        optimized_file_path=None,
        inter_op_num_threads=-1,
        intra_op_num_threads=-1,
    ):
        self.session = <void*>create_session()
        session_initialize(
            self.session,
            (optimized_file_path or "").encode('utf-8'),
            graph_optimization_level,
            1 if enable_cuda else 0,
            cuda_device_id,
            1 if set_denormal_as_zero else 0,
            intra_op_num_threads,
            inter_op_num_threads)
        if isinstance(filename, str):
            session_load_from_file(self.session, filename.encode('utf-8'))
        elif isinstance(filename, bytes):
            self._session_load_from_bytes(filename)
        else:
            raise TypeError(f"Unexpected type for filename {type(filename)}.")

    def __dealloc__(self):
        delete_session(self.session)

    def get_input_count(self):
        "Returns the number of inputs."
        return session_get_input_count(self.session)

    def get_output_count(self):
        "Returns the number of outputs."
        return session_get_output_count(self.session)

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def run_float_2(
        self,
        numpy.ndarray[numpy.float32_t] input1,
        numpy.ndarray[numpy.float32_t] input2
    ):
        """
        Runs the inference assuming the model has two inputs.
        """
        cdef OrtShape shapes[2]

        shapes[0].init(input1.ndim)
        for i in range(input1.ndim):
            shapes[0].set(i, input1.shape[i])

        shapes[1].init(input2.ndim)
        for i in range(input2.ndim):
            shapes[1].set(i, input2.shape[i])

        cdef numpy.ndarray[numpy.float32_t] value1 = numpy.ascontiguousarray(input1)
        cdef numpy.ndarray[numpy.float32_t] value2 = numpy.ascontiguousarray(input2)

        cdef OrtCpuValue in_values[2]
        in_values[0].init(value1.size(), 1, value1.data, <void*>0)
        in_values[1].init(value2.size(), 1, value2.data, <void*>0)

        cdef OrtShape out_shapes[10]
        cdef OrtCpuValue out_values[10]

        cdef size_t n_outputs = session_run(
            self.session, 2, shapes, in_values, 10, out_shapes, out_values)

        # onnxruntime does not implement the DLPack protocol through the C API
        # so we need to copy the data.

        cdef numpy.ndarray[numpy.int64_t, ndim=1] shape
        cdef numpy.ndarray[numpy.float32_t] t_float32
        res = list()
        for i in range(n_outputs):
            shape = numpy.empty(out_shapes[i].ndim(), dtype=numpy.int64)
            memcpy(shape.data, out_shapes[i].dims(), out_shapes[i].ndim() * 8)  # 8 = sizeof(int64)
            if out_values[i].elem_type() == 1:
                t_float32 = numpy.empty(shape=shape, dtype=numpy.float32)
                memcpy(t_float32.data, out_values[i].data(), out_values[i].size() * 4)  # 8 = sizeof(float32)
                out_values[i].free_ort_value()
                res.append(t_float32)
            else:
                raise NotImplementedError(
                    f"Unable to create a tensor for type {out_values[i].elem_type()}."
                )
        return res
