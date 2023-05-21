import numpy
cimport numpy
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
numpy.import_array()


# imports string
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


# imports vector
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


# imports DLDeviceType
cdef extern from "ortapi.h":

    enum DLDeviceType:
        kDLCPU = 1
        kDLCUDA = 2
        kDLCUDAHost = 3
        kDLOpenCL = 4
        kDLVulkan = 7
        kDLMetal = 8
        kDLVPI = 9
        kDLROCM = 10
        kDLROCMHost = 11
        kDLExtDev = 12
        kDLCUDAManaged = 13
        kDLOneAPI = 14
        kDLWebGPU = 15
        kDLHexagon = 16

# imports ONNXTensorElementDataType
cdef extern from "ortapi.h":

    enum ONNXTensorElementDataType:
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16

# imports OrtShape, InternalOrtCpuValue
cdef extern from "ortapi.h" namespace "ortapi":

    cdef cppclass OrtShape:
        OrtShape()
        void init(size_t)
        size_t ndim()
        void set(size_t i, int64_t dim)
        int64_t* dims()

    cdef cppclass InternalOrtCpuValue:
        InternalOrtCpuValue()
        void init(size_t size,
                  ONNXTensorElementDataType elem_type,
                  void* data,
                  void* ort_value)
        void free_ort_value()
        ONNXTensorElementDataType elem_type()
        size_t size()
        void* data()

    OrtShape* allocate_ort_shape(size_t n)
    InternalOrtCpuValue* allocate_ort_cpu_value(size_t n)
    void delete_ort_shape(OrtShape*)
    void delete_internal_ort_cpu_value(InternalOrtCpuValue*)
    void delete_ort_value(void*)

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
                       const InternalOrtCpuValue* values,
                       size_t max_outputs,
                       OrtShape** out_shapes,
                       InternalOrtCpuValue** out_values)

    void ort_value_get_shape_type(void* value,
                                  size_t& n_dims,
                                  ONNXTensorElementDataType& elem_type,
                                  int64_t* dims)


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


cdef class CyOrtShape:
    """
    Cython wrapper around an OrtShape.
    It contains one attribute of type `OrtShape`
    imported from C++.
    """
    cdef OrtShape shape

    def __init__(self):
        pass

    def set(self, shape):
        """
        Sets the shape.

        :param shape: tuple of ints
        """
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple of int not {type(shape)}.")
        self.shape.init(len(shape))
        for i in range(len(shape)):
            self.shape.set(i, shape[i])

    def __len__(self):
        "Returns the number of dimensions."
        return self.shape.ndim()

    def __getitem__(self, i):
        "Returns dimension i."
        return self.shape.dims()[i]

    def __setitem__(self, i, d):
        "Set dimension i."
        self.shape.set(i, d)

    def __repr__(self):
        "usual"
        nd = len(self)
        shape = tuple(self[i] for i in range(nd))
        return f"CyOrtShape({shape})"


cdef class CyOrtValue:
    """
    Wrapper around a OrtValue.
    """
    cdef void* ort_value_

    def __cinit__(self):
        self.ort_value_ = <void*>0

    cdef void set(self, void* ort_value):
        self.ort_value_ = ort_value

    @cython.initializedcheck(False)
    @property
    def shape_elem_type(self):
        """
        Returns the shape and the element type.
        """
        cdef int64_t dims[10]
        cdef size_t n_dims
        cdef ONNXTensorElementDataType elem_type
        ort_value_get_shape_type(self.ort_value_, n_dims, elem_type, dims)
        return tuple(dims[i] for i in range(n_dims)), elem_type

    def __dealloc__(self):
        if self.ort_value_ == <void*>0:
            raise RuntimeError("This instance of CyOrtValue was not properly initialized.")
        delete_ort_value(self.ort_value_)

    @staticmethod
    def from_dlpack(data):


    def __dlpack__(self, stream=None):
        
        
    def __dlpack_device__(self):
        

    


cdef class CyOrtSession:
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
        cdef OrtShape* shapes = allocate_ort_shape(len(inputs))
        cdef InternalOrtCpuValue* in_values = allocate_ort_cpu_value(len(inputs))
        cdef numpy.ndarray value

        values = []
        for n in range(len(inputs)):
            shapes[n].init(inputs[n].ndim)
            for i in range(inputs[n].ndim):
                shapes[n].set(i, inputs[n].shape[i])
            value = numpy.ascontiguousarray(inputs[n])
            in_values[n].init(
                value.size,
                CyOrtSession._onnx_types[value.dtype],
                value.data,
                <void*>0
            )
            # otherwise the pointer might be destroyed by the garbage collector
            values.append(value)

        cdef OrtShape* out_shapes
        cdef InternalOrtCpuValue* out_values

        cdef size_t n_outputs = session_run(
            self.session, len(inputs), shapes, in_values, 10, &out_shapes, &out_values)

        delete_ort_shape(shapes)
        delete_internal_ort_cpu_value(in_values)

        # onnxruntime does not implement the DLPack protocol through the C API.
        # DLPack protocol should be used.

        cdef numpy.ndarray[numpy.int64_t, ndim=1] shape
        cdef numpy.ndarray tout
        res = list()
        for i in range(n_outputs):
            shape = numpy.empty(out_shapes[i].ndim(), dtype=numpy.int64)
            memcpy(shape.data,
                   out_shapes[i].dims(),
                   out_shapes[i].ndim() * 8)  # 8 = sizeof(int64)
            tout = numpy.empty(
                shape=shape,
                dtype=CyOrtSession._dtypes[out_values[i].elem_type()]
            )
            memcpy(tout.data,
                   out_values[i].data(),
                   out_values[i].size() * ElementSizeI(out_values[i].elem_type()))
            out_values[i].free_ort_value()
            res.append(tout)

        delete_ort_shape(out_shapes)
        delete_internal_ort_cpu_value(out_values)
        return res
