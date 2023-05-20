import os
import platform
import textwrap
from typing import List

_ort_ext_libs = []


def get_ort_ext_libs() -> List[str]:
    """
    Returns the list of libraries implementing new simple
    :epkg:`onnxruntime` kernels implemented for the
    :epkg:`CPUExecutionProvider`.
    """
    global _ort_ext_libs
    if len(_ort_ext_libs) == 0:
        if platform.system() == "Windows":
            ext = ".dll"
        elif platform.system() == "Darwin":
            ext = ".dylib"
        else:
            ext = ".so"
        this = os.path.abspath(os.path.dirname(__file__))
        files = os.listdir(this)
        res = []
        for name in files:
            e = os.path.splitext(name)[-1]
            if e == ext and "ortops" in name:
                res.append(os.path.join(this, name))
        if len(res) == 0:
            raise RuntimeError(
                f"Unable to find any kernel library with ext={ext!r} "
                f"in {this!r} among {files}."
            )
        _ort_ext_libs = res
    return _ort_ext_libs


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
    onnx_extented.ortops.tutorial.cpu.MyCustomOp
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors.

    **Provider**
    
    CPUExecutionProvider
    
    **Inputs**
    
    * X (T): tensor of type T
    * Y (T): tensor of type T

    **Outputs**

    * Z (T): addition of X, Y

    **Constraints**

    * T: float
    """,
                """
    onnx_extented.ortops.tutorial.cpu.MyCustomOpWithAttributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors + a constant equal to
    `cst = att_float + att_int64 + att_string[0] + att_tensot[0]`.

    **Provider**
    
    CPUExecutionProvider
    
    **Attributes**

    * att_float: a float
    * att_int64: an integer
    * att_tensor: a tensor of any type and shape
    * att_string: a string
    
    **Inputs**
    
    * X (T): tensor of type T
    * Y (T): tensor of type T

    **Outputs**

    * Z (T): addition of X, Y + cst

    **Constraints**

    * T: float
    """,
            ],
        )
    )
