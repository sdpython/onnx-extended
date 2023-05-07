import numpy
cimport numpy
cimport cython
numpy.import_array()


cdef extern from "ortapi.h" namespace "ortapi":

    void* create_session();
    void delete_session(void*);
    size_t get_input_count(void*);
    size_t get_output_count(void*);
    void session_load_from_file(void*, const char* filename);

    # void* ort_run_inference(void* inst, int n_inputs, const char** input_names, void* ort_values, int& n_outputs);

    # https://stackoverflow.com/questions/46510654/using-pycapsule-in-cython
    # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/squeezenet/main.cpp




cdef class OrtSession:
    """
    Wrapper around `onnxruntime C API
    <https://onnxruntime.ai/docs/api/c/>`_.
    """

    cdef void* session;

    def __init__(self, str filename):
        self.session = <void*>create_session();
        session_load_from_file(self.session, filename.encode('utf-8'));

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
def run_inference(inst, input_names, values):
    """
    Runs inference and returns the results.
    """
    raise NotImplementedError()

