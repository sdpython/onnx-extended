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
    onnx_extended.ortops.optim.cuda.AddMul
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Two consecutive element-wise Add, Mul assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * (A+B)*C (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.AddAdd
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Two consecutive element-wise addition assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * A+B+C (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.AddAddAdd
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Three consecutive element-wise addition assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T
    * D (T): tensor of type T

    **Outputs**

    * A+B+C+D (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.AddSharedInput
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Parallel Additions with one common input.
    Support for Broadcast is limited (only the first dimenions).
    
    Computes A + B, A + C.

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * A+B (T): element-wise
    * A+C (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.MulAdd
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Two consecutive element-wise Mul, Add assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * A*B+C (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.MulMul
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Two consecutive element-wise multiplication assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * ABC (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.MulMulMul
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Two consecutive element-wise multiplication assuming
    all tensors have the same shape (no broadcast).

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T
    * D (T): tensor of type T

    **Outputs**

    * ABCD (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.MulSoftmax
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    MulSoftmax, equivalent to Mul(X, Softmax(X))

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T): tensor
    
    **Outputs**

    * Z (T): result

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.MulSharedInput
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Parallel Multiplications with one common input.
    Support for Broadcast is limited (only the first dimenions).
    
    Computes A * B, A * C.

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * A (T): tensor of type T
    * B (T): tensor of type T
    * C (T): tensor of type T

    **Outputs**

    * A*B (T): element-wise
    * A*C (T): element-wise

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.NegXplus1
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    1 - X

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T): tensor of type T
    
    **Outputs**

    * Z (T): result

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.ReplaceZero
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    MulSoftmax, equivalent to Where(X == 0, cst, X)

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T): tensor of type T
    
    **Outputs**

    * Z (T): updated tensor

    **Constraints**

    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.Rotary
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Rotary, equivalent to (side=="RIGHT")
    
    * Split(X, axis=-1) -> X1, X2
    * Concat(-X2, X1)

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T): tensor
    * splits (I): split size on the last dimension
    
    Only splitting in half is implemented.

    **Outputs**

    * Z (T): result

    **Constraints**

    * T: float, float16
    * I: int64
    """,
                """
    onnx_extended.ortops.optim.cuda.ScatterNDOfShape
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ConstantOfShape + ScatterND

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * shape (I): tensor of type I
    * indices (I): tensor of type I
    * updates (T): tensor of type T

    **Outputs**

    * Z (T): updated tensor

    **Constraints**

    * I: int64
    * T: float, float16
    """,
                """
    onnx_extended.ortops.optim.cuda.Transpose2DCast16
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Transposes a 2D matrix the cast it into float16.
    
    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T1): tensor
    
    Only splitting in half is implemented.

    **Outputs**

    * Z (T2): result

    **Constraints**

    * T1: float32
    * T2: float16
    """,
                """
    onnx_extended.ortops.optim.cuda.Transpose2DCast32
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Transposes a 2D matrix the cast it into float32.
    
    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * X (T1): tensor
    
    Only splitting in half is implemented.

    **Outputs**

    * Z (T2): result

    **Constraints**

    * T1: float16
    * T2: float32
    """,
                """
    onnx_extended.ortops.optim.cuda.TriMatrix
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Creates a matrix.
    
    ::
    
        mat[i < j] = upper
        mat[i == j] = diag
        mat[i > j] = lower

    **Provider**
    
    CUDAExecutionProvider
    
    **Inputs**
    
    * shape (I): tensor of type I
    * cst (T): lower, diag, upper values

    **Outputs**

    * Z (T): matrix

    **Constraints**

    * I: int64
    * T: float, float16
    """,
            ],
        )
    )
