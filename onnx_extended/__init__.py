# coding: utf-8
"""
More operator for onnx reference implementation.
"""

__version__ = "0.1.0"
__author__ = "Xavier Dupré"


def has_cuda():
    """
    Tells if cuda is available.
    """
    from ._config import HAS_CUDA

    return HAS_CUDA == 1


def cuda_version():
    """
    Tells which version of CUDA was used to build the CUDA extensions.
    """
    if not has_cuda():
        raise RuntimeError("CUDA extensions are not available.")
    from ._config import CUDA_VERSION

    return CUDA_VERSION
