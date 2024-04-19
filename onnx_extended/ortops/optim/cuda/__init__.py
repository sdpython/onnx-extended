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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T

    **Outputs**

    * (A+B)*C (T): element-wise matrix multiplication

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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T

    **Outputs**

    * A+B+C (T): element-wise matrix multiplication

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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T
    * D: tensor of type T

    **Outputs**

    * A+B+C+D (T): element-wise matrix multiplication

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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T

    **Outputs**

    * A*B+C (T): element-wise matrix multiplication

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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T

    **Outputs**

    * ABC (T): element-wise matrix multiplication

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
    
    * A: tensor of type T
    * B: tensor of type T
    * C: tensor of type T
    * D: tensor of type T

    **Outputs**

    * ABCD (T): element-wise matrix multiplication

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
    
    * X (T): tensor of type I
    
    Only splitting in half is implemented.

    **Outputs**

    * Z (T): updates tensor

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
    
    * X (T): tensor of type I
    * splits (I): split size on the last dimension
    
    Only splitting in half is implemented.

    **Outputs**

    * Z (T): updates tensor

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
    
    * shape: tensor of type I
    * indices: tensor of type I
    * updates: tensor of type I

    **Outputs**

    * Z (T): updates tensor

    **Constraints**

    * I: int64
    * T: float, float16
    """,
            ],
        )
    )
