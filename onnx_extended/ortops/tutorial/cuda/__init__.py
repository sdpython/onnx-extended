import os
import textwrap
from typing import List
from ... import _get_ort_ext_libs


def get_ort_ext_libs() -> List[str]:
    """
    Returns the list of libraries implementing new simple
    :epkg:`onnxruntime` kernels implemented for the
    :epkg:`CUDAExecutionProvider`.
    """
    libs = _get_ort_ext_libs(os.path.dirname(__file__))
    return [lib for lib in libs if "cuda_cuda" not in lib]


def documentation() -> List[str]:
    """
    Returns a list of rst string documenting every implemented kernels
    in this subfolder.
    """
    return list(
        map(
            textwrap.dedent,
            [
                """
    onnx_extented.ortops.tutorial.cuda.CustomGemm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It calls CUDA library for Gemm :math:`\\alpha A B + \\beta C`.

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T
    * D (T): tensor of type T
    * E (T): tensor of type T

    **Outputs**

    * Z (T): :math:`\\alpha A B + \\beta C`

    **Constraints**

    * T: float, float16, bfloat16
    """
            ],
        )
    )
