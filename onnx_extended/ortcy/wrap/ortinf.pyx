import numpy
cimport numpy
cimport cython
numpy.import_array()


# cdef extern from "ortapi.h" namespace "ortapi":
#     void* ort_create_session(const char* filename, void* sess_options);

# cdef extern from "ortapi.h" namespace "ortapi":
#     void* ort_run_inference(void* inst, int n_inputs, const char** input_names, void* ort_values, int& n_outputs);

# https://stackoverflow.com/questions/46510654/using-pycapsule-in-cython

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
