# coding: utf-8
"""
More operators for onnx reference implementation and onnxruntime.
Experimentation with openmp, CUDA.
"""

__version__ = "0.2.3"
__author__ = "Xavier Dupré"


def check_installation(
    ortops: bool = True, ortcy: bool = False, val: bool = True, verbose: bool = False
):
    """
    Quickly checks the installation works.

    :param ortops: checks that custom ops on CPU are working
    :param ortcy: checks that OrtSession is working (cython bindings of onnxruntime)
    :param val: checks that a couple of functions
        in submodule validation are working
    """
    assert isinstance(get_cxx_flags(), str)
    import datetime
    import warnings

    def local_print(msg):
        t = datetime.datetime.now().time()
        print(msg.replace("[check_installation]", f"[check_installation] {t}"))

    if val:
        if verbose:
            local_print("[check_installation] import numpy")
        import numpy

        if verbose:
            local_print("[check_installation] import onnx-extended")
        from onnx_extended.validation.cython.fp8 import cast_float32_to_e4m3fn

        a = ((numpy.arange(10).astype(numpy.float32) - 5) / 10).astype(numpy.float32)
        if verbose:
            local_print("[check_installation] cast_float32_to_e4m3fn")
        f8 = cast_float32_to_e4m3fn(a)
        assert a.shape == f8.shape

    with warnings.catch_warnings(record=False):
        if verbose:
            local_print("[check_installation] import onnx, numpy")
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

        if verbose:
            local_print("[check_installation] create a simple onnx model")

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
        a = numpy.random.randn(2, 2).astype(numpy.float32)
        b = numpy.random.randn(2, 2).astype(numpy.float32)
        feeds = {"X": a, "A": b}

        if ortops:
            if verbose:
                local_print("[check_installation] import onnxruntime")
            from onnxruntime import InferenceSession, SessionOptions

            if verbose:
                local_print("[check_installation] import onnx-extended")
            from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

            r = get_ort_ext_libs()
            if verbose:
                local_print(
                    f"[check_installation] get_ort_ext_libs()={get_ort_ext_libs()!r}"
                )
            opts = SessionOptions()
            opts.register_custom_ops_library(r[0])
            if verbose:
                local_print("[check_installation] create session")
            sess = InferenceSession(
                onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
            )
            if verbose:
                local_print("[check_installation] run session")
            got = sess.run(None, feeds)[0]
            if verbose:
                local_print("[check_installation] check shapes")
            assert (a + b).shape == got.shape

        if ortcy:
            if verbose:
                local_print("[check_installation] import onnx-extended")
            from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs
            from onnx_extended.ortcy.wrap.ortinf import OrtSession

            r = get_ort_ext_libs()
            if verbose:
                local_print(
                    f"[check_installation] get_ort_ext_libs()={get_ort_ext_libs()!r}"
                )
            if verbose:
                local_print("[check_installation] create OrtSession")
            session = OrtSession(onnx_model.SerializeToString(), custom_libs=r)
            if verbose:
                local_print("[check_installation] run OrtSession")
            got = session.run([a, b])
            if verbose:
                local_print("[check_installation] check shapes")
            assert (a + b).shape == got[0].shape

        if verbose:
            local_print("[check_installation] done")


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
