import os
import textwrap
from typing import List
from ... import _get_ort_ext_libs


def get_ort_ext_libs() -> List[str]:
    """
    Returns the list of libraries implementing new simple
    :epkg:`onnxruntime` kernels implemented for the
    :epkg:`CPUExecutionProvider`.
    """
    return _get_ort_ext_libs(os.path.dirname(__file__))


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
    onnx_extended.ortops.tutorial.cpu.CustomGemmFloat
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements operator Gemm for float type.
    Operator CustomGemmFloat16 implements the same for the float 16.
    CustomGemmFloat8E4M3FN allows Float8E4M3FN as inputs and floats
    as outputs.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    * to: quantized type

    **Inputs**
    
    * A (T1): tensor of type T
    * B (T1): tensor of type T
    * C (T2): tensor of type T (optional)
    * scaleA (TF): scale for A (optional)
    * scaleB (TF): scale for B (optional)
    * scale (TF): scale for the result (optional)

    **Outputs**

    * Y (T2): result of Gemm

    **Constraints**

    * T1: float, float 16 or Float8E4M3FN
    * T2: float or float 16
    * TF: float
    """,
                """
    onnx_extended.ortops.tutorial.cpu.DynamicQuantizeLinear
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements DynamicQuantizeLinear opset 20.

    **Provider**
    
    CPUExecutionProvider
    
    **Attributes**

    * to: quantized type

    **Inputs**
    
    * X (T1): tensor of type T

    **Outputs**

    * Y (T2): quantized X
    * scale (TS): scale
    * Y (T2): zero point

    **Constraints**

    * T1: float, float 16
    * TS: float
    * T2: int8, uint8, float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
    """,
                """
    onnx_extended.ortops.tutorial.cpu.MyCustomOp
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
    onnx_extended.ortops.tutorial.cpu.MyCustomOpWithAttributes
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
