"""
More operators for onnx reference implementation and onnxruntime.
Experimentation with openmp, CUDA.
"""

__version__ = "0.4.0"
__author__ = "Xavier Dupré"


def _check_installation_ortcy(onnx_model, verbose):
    import datetime

    def local_print(msg):
        t = datetime.datetime.now().time()
        print(
            msg.replace("[check_installation_ortcy]", f"[check_installation_ortcy] {t}")
        )

    if verbose:
        local_print("[check_installation_ortcy] --begin")
    import gc
    import numpy

    a = numpy.random.randn(2, 2).astype(numpy.float32)
    b = numpy.random.randn(2, 2).astype(numpy.float32)

    if verbose:
        local_print("[check_installation_ortcy] import onnx-extended")
    try:
        from onnx_extended.ortcy.wrap.ortinf import OrtSession
    except ImportError as e:
        import os
        from onnx_extended.ortcy.wrap import __file__ as cyfile

        this = os.path.dirname(cyfile)
        files = os.listdir(this)
        if "libonnxruntime.so.1.19.2" in files:
            if verbose:
                local_print(
                    "[check_installation_ortcy] weird issue as the "
                    f"so is in onnx_extended.ortcy.wrap: {files}."
                )
            return
        raise ImportError(
            f"Unable to import OrtSession, "
            f"content in onnx_extended.ortcy.wrap is {files}."
        ) from e
    from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

    r = get_ort_ext_libs()
    if verbose:
        local_print(
            f"[check_installation_ortcy] get_ort_ext_libs()={get_ort_ext_libs()!r}"
        )
    if verbose:
        local_print("[check_installation_ortcy] create OrtSession")
    session = OrtSession(onnx_model.SerializeToString(), custom_libs=r)
    if verbose:
        local_print("[check_installation_ortcy] run OrtSession")
    got = session.run([a, b])
    if verbose:
        local_print("[check_installation_ortcy] second run")
    got = session.run([a, b])
    if verbose:
        local_print("[check_installation_ortcy] check shapes")
    assert (a + b).shape == got[0].shape
    if verbose:
        local_print("[check_installation_ortcy] gc")
    gc.collect()
    if verbose:
        local_print("[check_installation_ortcy] --done")


def _check_installation_ortops(onnx_model, verbose):
    import datetime

    def local_print(msg):
        t = datetime.datetime.now().time()
        print(
            msg.replace(
                "[check_installation_ortops]", f"[check_installation_ortops] {t}"
            )
        )

    if verbose:
        local_print("[check_installation_ortops] --begin")

    import gc
    import numpy

    a = numpy.random.randn(2, 2).astype(numpy.float32)
    b = numpy.random.randn(2, 2).astype(numpy.float32)
    feeds = {"X": a, "A": b}

    if verbose:
        local_print("[check_installation_ortops] import onnxruntime")
    from onnxruntime import InferenceSession, SessionOptions

    if verbose:
        local_print("[check_installation_ortops] import onnx-extended")
    from onnx_extended.ortops.tutorial.cpu import get_ort_ext_libs

    r = get_ort_ext_libs()
    if verbose:
        local_print(
            f"[check_installation_ortops] get_ort_ext_libs()={get_ort_ext_libs()!r}"
        )
    opts = SessionOptions()
    opts.register_custom_ops_library(r[0])
    if verbose:
        local_print("[check_installation_ortops] create session")
    sess = InferenceSession(
        onnx_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    if verbose:
        local_print("[check_installation_ortops] run session")
    if verbose:
        local_print("[check_installation_ortops] second run")
    got = sess.run(None, feeds)[0]
    got = sess.run(None, feeds)[0]
    if verbose:
        local_print("[check_installation_ortops] check shapes")
    assert (a + b).shape == got.shape
    if verbose:
        local_print("[check_installation_ortcy] gc")
    gc.collect()
    if verbose:
        local_print("[check_installation_ortops] --done")


def check_installation(
    ortops: bool = False, ortcy: bool = False, val: bool = False, verbose: bool = False
):
    """
    Quickly checks the installation works.

    :param ortops: checks that custom ops on CPU are working
    :param ortcy: checks that OrtSession is working (cython bindings of onnxruntime)
    :param val: checks that a couple of functions
        in submodule validation are working
    :param verbose: prints out which verifications is being processed
    """
    import datetime

    def local_print(msg):
        t = datetime.datetime.now().time()
        print(msg.replace("[check_installation]", f"[check_installation] {t}"))

    if verbose:
        local_print("[check_installation] --begin")
    assert isinstance(get_cxx_flags(), str)
    import warnings

    if val:
        if verbose:
            local_print("[check_installation] --val")
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
        if verbose:
            local_print("[check_installation] --done")

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
            "MyCustomOp", ["X", "A"], ["Y"], domain="onnx_extended.ortops.tutorial.cpu"
        )
        graph = make_graph([node1], "lr", [X, A], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[make_opsetid("onnx_extended.ortops.tutorial.cpu", 1)],
            ir_version=8,
        )
        check_model(onnx_model)

        if ortcy:
            _check_installation_ortcy(onnx_model, verbose)

        if ortops:
            _check_installation_ortops(onnx_model, verbose)

        if verbose:
            local_print("[check_installation] --done")


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
    assert has_cuda(), "CUDA extensions are not available."
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


def ort_version() -> str:
    """
    Tells which version of onnxruntime it was built with.
    """
    from ._config import ORT_VERSION

    return ORT_VERSION


def ort_version_int() -> tuple:
    """
    Tells which version of onnxruntime was used to build
    the onnxruntime extensions. It returns `(0, 0)`
    if onnxruntime is not present.
    """
    from ._config import ORT_VERSION

    if not isinstance(ORT_VERSION, str):
        return tuple()

    spl = ORT_VERSION.split(".")
    return tuple(map(int, spl))
