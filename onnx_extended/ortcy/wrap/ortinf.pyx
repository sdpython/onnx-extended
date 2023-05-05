import numpy
cimport numpy
cimport cython
numpy.import_array()


cdef extern from "ortapi.h" namespace "ortapi":

    void *ort_create_env();
    void ort_delete_env(void*);

    void* ort_create_session_options();
    void ort_delete_session_options(void*);

    void* ort_create_session(const char* filename, void* env, void* sess_options);
    void ort_delete_session(void *session);

    # void* ort_run_inference(void* inst, int n_inputs, const char** input_names, void* ort_values, int& n_outputs);

    # https://stackoverflow.com/questions/46510654/using-pycapsule-in-cython
    # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/squeezenet/main.cpp


cdef class OrtSessionOptions:

    cdef void* pointer

    def __init__(self):
        self.pointer = ort_create_session_options()

    def __dealloc__(self):
        ort_delete_session_options(self.pointer)


cdef class OrtInference:

    cdef void* pointer
    cdef void* env;

    def __init__(self, filename, OrtSessionOptions sess_options):
        self.env = ort_create_env();
        self.pointer = ort_create_session(filename, self.env, sess_options.pointer)

    def __dealloc__(self):
        ort_delete_session(self.pointer)
        ort_delete_env(self.env)


def create_inference(filename, sess_options):
    """
    Creates an object InferenceSession.

    :param filename: filename
    :param sess_options: session options
    :return: capsule
    """
    raise NotImplementedError()


def create_session_options():
    """
    Creates an instance of SessionOptions in a capsule.

    """
    raise NotImplementedError()


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def run_inference_float(inst, input_name, value):
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

