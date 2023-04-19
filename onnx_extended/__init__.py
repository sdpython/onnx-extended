# coding: utf-8
"""
More operator for onnx reference implementation.
"""

__version__ = "0.1.0"
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


def compiled_with_cuda():
    """
    Checks it was compiled with CUDA.
    """
    try:
        from .validation.cuda import cuda_example_py

        return cuda_example_py is not None
    except ImportError:
        return False
