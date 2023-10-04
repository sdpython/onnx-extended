# coding: utf-8
"""
More operators for onnx reference implementation and onnxruntime.
Experimentation with openmp, CUDA.
"""

__version__ = "0.2.3"
__author__ = "Xavier Dupré"


def check_installation():
    """
    Checks the installation works.
    """
    assert isinstance(get_cxx_flags(), str)
    import warnings

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        import numpy
        from onnx import TensorProto
        from onnx.helper import (
            make_model,
            make_node,
            make_graph,
            make_tensor_value_info,
            make_opsetid,
        )
        from onnx.checker import check_model
        from onnxruntime import InferenceSession, SessionOptions
        from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "MyCustomOp", ["X", "A"], ["Y"], domain="onnx_extented.ortops.tutorial.cpu"
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[make_opsetid("onnx_extented.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        r = get_ort_ext_libs()
        opts = SessionOptions()
        opts.register_custom_ops_library(r[0])
        sess = InferenceSession(
            onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        a = numpy.random.randn(2, 2).astype(numpy.float32)
        b = numpy.random.randn(2, 2).astype(numpy.float32)
        feeds = {"X": a, "A": b}
        got = sess.run(None, feeds)[0]
        assert (a + b).shape == got.shape


def has_cuda() -> bool:
    """
    Tells if cuda is available.
    """
    from ._config import HAS_CUDA

    return HAS_CUDA == 1


def cuda_version() -> str:
    """
    Tells which version of CUDA was used to build the CUDA extensions.
    """
    if not has_cuda():
        raise RuntimeError("CUDA extensions are not available.")
    from ._config import CUDA_VERSION

    return CUDA_VERSION


def cuda_version_int() -> tuple:
    """
    Tells which version of CUDA was used to build the CUDA extensions.
    It returns `(0, 0)` if CUDA is not present.
    """
    if not has_cuda():
        return (0, 0)
    from ._config import CUDA_VERSION

    if not isinstance(CUDA_VERSION, str):
        return tuple()

    spl = CUDA_VERSION.split(".")
    return tuple(map(int, spl))


def compiled_with_cuda() -> bool:
    """
    Checks it was compiled with CUDA.
    """
    try:
        from .validation.cuda import cuda_example_py

        return cuda_example_py is not None
    except ImportError:
        return False


def get_cxx_flags() -> str:
    """
    Returns `CXX_FLAGS`.
    """
    from ._config import CXX_FLAGS

    return CXX_FLAGS


def get_stdcpp() -> int:
    """
    Returns `CMAKE_CXX_STANDARD`.
    """
    from ._config import CMAKE_CXX_STANDARD

    return CMAKE_CXX_STANDARD
