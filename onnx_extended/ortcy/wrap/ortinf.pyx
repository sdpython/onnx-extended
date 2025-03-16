import numpy
cimport numpy as cnumpy
cimport cython
from libc.stdint cimport int64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from cpython cimport Py_buffer
from cpython.buffer cimport (
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_SIMPLE,
)
# numpy.import_array()


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

    cdef int ort_c_api_supported_version()

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

    size_t ElementSizeI(int elem_type)
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
                            int inter_op_num_threads,
                            char** custom_libs)
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


def get_ort_c_api_supported_version():
    """
    Returns the supported version of onnxruntime C API.
    """
    return ort_c_api_supported_version()


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

    .. versionadded:: 0.2.0
    """

    # see https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3#L485
    _dtypes = {
        1: numpy.float32,
        2: numpy.uint8,
        3: numpy.int8,
        4: numpy.uint16,
        5: numpy.int16,
        6: numpy.int32,
        7: numpy.int64,
        # 8: numpy.str_,
        9: numpy.bool_,
        10: numpy.float16,
        11: numpy.float64,
        12: numpy.uint32,
        13: numpy.uint64,
    }

    _onnx_types = {
        numpy.float32: 1,
        numpy.dtype("float32"): 1,
        numpy.uint8: 2,
        numpy.dtype("uint8"): 2,
        numpy.int8: 3,
        numpy.dtype("int8"): 3,
        numpy.uint16: 4,
        numpy.dtype("uint16"): 4,
        numpy.int16: 5,
        numpy.dtype("int16"): 5,
        numpy.int32: 6,
        numpy.dtype("int32"): 6,
        numpy.int64: 7,
        numpy.dtype("int64"): 7,
        # 8: numpy.str_,
        # numpy.dtype("str_"): 8,
        numpy.bool_: 9,
        numpy.dtype("bool_"): 9,
        numpy.float16: 10,
        numpy.dtype("float16"): 10,
        numpy.float64: 11,
        numpy.dtype("float64"): 11,
        numpy.uint32: 12,
        numpy.dtype("uint32"): 12,
        numpy.uint64: 13,
        numpy.dtype("uint64"): 13,
    }

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
        custom_libs=None,
    ):
        cdef char** c_custom_libs = <char**> 0

        custom_libs_encoded = (
            None if custom_libs is None else
            [c.encode("utf-8") for c in custom_libs]
        )
        if custom_libs is not None:
            c_custom_libs = <char**> malloc((len(custom_libs) + 1) * sizeof(char*))
            for i in range(len(custom_libs)):
                c_custom_libs[i] = <char*>custom_libs_encoded[i]
            c_custom_libs[len(custom_libs)] = <char*> 0
        opt_file_path = (optimized_file_path or "").encode('utf-8')

        self.session = <void*>create_session()
        session_initialize(
            self.session,
            opt_file_path,
            graph_optimization_level,
            1 if enable_cuda else 0,
            cuda_device_id,
            1 if set_denormal_as_zero else 0,
            intra_op_num_threads,
            inter_op_num_threads,
            c_custom_libs)

        if c_custom_libs != (<char**> 0):
            free(c_custom_libs)

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
    def run(
        self,
        list inputs,
    ):
        """
        Runs the inference.
        The number of inputs and outputs must not exceed 10.
        """
        if len(inputs) > 10:
            raise RuntimeError(
                f"This function does not work with more than "
                f"10 inputs ({len(inputs)})."
            )
        cdef OrtShape shapes[10]
        cdef OrtCpuValue in_values[10]
        cdef cnumpy.ndarray value

        values = []
        for n in range(len(inputs)):
            shapes[n].init(inputs[n].ndim)
            for i in range(inputs[n].ndim):
                shapes[n].set(i, inputs[n].shape[i])
            value = numpy.ascontiguousarray(inputs[n])
            in_values[n].init(
                value.size,
                OrtSession._onnx_types[value.dtype],
                value.data,
                <void*>0
            )
            # otherwise the pointer might be destroyed by the garbage collector
            values.append(value)

        cdef OrtShape out_shapes[10]
        cdef OrtCpuValue out_values[10]

        cdef size_t n_outputs = session_run(
            self.session, len(inputs), shapes, in_values, 10, out_shapes, out_values)

        # onnxruntime does not implement the DLPack protocol through the C API.
        # DLPack protocol should be used.

        cdef cnumpy.ndarray[cnumpy.int64_t, ndim=1] shape
        cdef cnumpy.ndarray tout
        res = list()
        for i in range(n_outputs):
            shape = numpy.empty(out_shapes[i].ndim(), dtype=numpy.int64)
            memcpy(shape.data,
                   out_shapes[i].dims(),
                   out_shapes[i].ndim() * 8)  # 8 = sizeof(int64)
            tout = numpy.empty(
                shape=shape,
                dtype=OrtSession._dtypes[out_values[i].elem_type()]
            )
            memcpy(tout.data,
                   out_values[i].data(),
                   out_values[i].size() * ElementSizeI(out_values[i].elem_type()))
            out_values[i].free_ort_value()
            res.append(tout)
        return res

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def run_1_1(
        self,
        cnumpy.ndarray input1,
    ):
        """
        Runs the inference assuming the model has one input and one output.
        """
        cdef OrtShape shapes[1]

        shapes[0].init(input1.ndim)
        for i in range(input1.ndim):
            shapes[0].set(i, input1.shape[i])

        cdef cnumpy.ndarray value1 = numpy.ascontiguousarray(input1)
        cdef OrtCpuValue in_values[1]

        in_values[0].init(
            value1.size,
            OrtSession._onnx_types[value1.dtype],
            value1.data,
            <void*>0
        )

        cdef OrtShape out_shapes[1]
        cdef OrtCpuValue out_values[1]

        cdef size_t n_outputs = session_run(
            self.session, 1, shapes, in_values, 1, out_shapes, out_values)
        if n_outputs != 1:
            raise RuntimeError(f"Expecting 1 output not {n_outputs}.")

        # onnxruntime does not implement the DLPack protocol through the C API.
        # DLPack protocol should be used.

        cdef cnumpy.ndarray[cnumpy.int64_t, ndim=1] shape = numpy.empty(
            out_shapes[0].ndim(), dtype=numpy.int64)
        memcpy(shape.data,
               out_shapes[0].dims(),
               out_shapes[0].ndim() * 8)  # 8 = sizeof(int64)
        cdef cnumpy.ndarray tout = numpy.empty(
            shape=shape,
            dtype=OrtSession._dtypes[out_values[0].elem_type()]
        )
        memcpy(tout.data,
               out_values[0].data(),
               out_values[0].size() * ElementSizeI(out_values[0].elem_type()))
        out_values[0].free_ort_value()
        return tout

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def run_2(
        self,
        cnumpy.ndarray input1,
        cnumpy.ndarray input2
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

        cdef cnumpy.ndarray value1 = numpy.ascontiguousarray(input1)
        cdef cnumpy.ndarray value2 = numpy.ascontiguousarray(input2)

        cdef OrtCpuValue in_values[2]
        in_values[0].init(
            value1.size,
            OrtSession._onnx_types[value1.dtype],
            value1.data,
            <void*>0
        )
        in_values[1].init(
            value2.size,
            OrtSession._onnx_types[value2.dtype],
            value2.data,
            <void*>0
        )

        cdef OrtShape out_shapes[10]
        cdef OrtCpuValue out_values[10]

        cdef size_t n_outputs = session_run(
            self.session, 2, shapes, in_values, 10, out_shapes, out_values)

        # onnxruntime does not implement the DLPack protocol through the C API.
        # DLPack protocol should be used.

        cdef cnumpy.ndarray[cnumpy.int64_t, ndim=1] shape
        cdef cnumpy.ndarray tout
        res = list()
        for i in range(n_outputs):
            shape = numpy.empty(out_shapes[i].ndim(), dtype=numpy.int64)
            memcpy(shape.data,
                   out_shapes[i].dims(),
                   out_shapes[i].ndim() * 8)  # 8 = sizeof(int64)
            tout = numpy.empty(
                shape=shape,
                dtype=OrtSession._dtypes[out_values[i].elem_type()]
            )
            memcpy(tout.data,
                   out_values[i].data(),
                   out_values[i].size() * ElementSizeI(out_values[i].elem_type()))
            out_values[i].free_ort_value()
            res.append(tout)
        return res
