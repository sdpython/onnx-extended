# coding: utf-8
"""
More operators for onnx reference implementation.
Experimentation with openmp, CUDA.
"""

__version__ = "0.2.0"
__author__ = "Xavier Dupré"


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
    """
    if not has_cuda():
        raise RuntimeError("CUDA extensions are not available.")
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
    Returns `CXX_FLAGS`.
    """
    import re

    reg = re.compile("-std=c[+][+]([0-9]+)")
    f = reg.findall(get_cxx_flags())
    if len(f) == 0:
        raise ValueError(f"Unable to extract c++ version from {get_cxx_flags()!r}.")
    return int(f[0])
