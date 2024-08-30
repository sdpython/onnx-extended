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
    onnx_extended.ortops.option.cpu.DenseToSparse
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Converts a dense tensor into a sparse one.
    All null values are skipped.

    **Provider**

    CPUExecutionProvider

    **Inputs**

    * X (T): 2D tensor

    **Outputs**

    * Y (T): 1D tensor

    **Constraints**

    * T: float
    """,
                """
    onnx_extended.ortops.option.cpu.SparseToDense
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Converts a spadenserse tensor into a sparse one.
    All missing values are replaced by 0.

    **Provider**

    CPUExecutionProvider

    **Inputs**

    * X (T): 1D tensor

    **Outputs**

    * Y (T): 2D tensor

    **Constraints**

    * T: float
    """,
                """
    onnx_extended.ortops.option.cpu.TfIdfVectorizer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements TfIdfVectorizer.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    See `onnx TfIdfVectorizer
    <https://onnx.ai/onnx/operators/onnx_aionnxml_TfIdfVectorizer.html>`_.
    The implementation does not support string labels. It is adding one attribute.

    * sparse: INT64, default is 0, the output and the computation are sparse,
      see

    **Inputs**

    * X (T1): tensor of type T1

    **Outputs**

    * label (T3): labels of type T3
    * Y (T2): probabilities of type T2

    **Constraints**

    * T1: float, double
    * T2: float, double
    * T3: int64
    """,
                """
    onnx_extended.ortops.option.cpu.TreeEnsembleClassifier
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    See `onnx TreeEnsembleClassifier
    <https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html>`_.
    The implementation does not support string labels.
    The only change:

    nodes_modes: string contenation with `,`

    **Inputs**

    * X (T1): tensor of type T1

    **Outputs**

    * label (T3): labels of type T3
    * Y (T2): probabilities of type T2

    **Constraints**

    * T1: float, double
    * T2: float, double
    * T3: int64
    """,
                """
    onnx_extended.ortops.option.cpu.TreeEnsembleClassifierSparse
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    See `onnx TreeEnsembleClassifier
    <https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html>`_.
    The implementation does not support string labels.
    The only change:

    nodes_modes: string contenation with `,`

    **Inputs**

    * X (T1): tensor of type T1 (sparse)

    **Outputs**

    * label (T3): labels of type T3
    * Y (T2): probabilities of type T2

    **Constraints**

    * T1: float, double
    * T2: float, double
    * T3: int64
    """,
                """
    onnx_extended.ortops.option.cpu.TreeEnsembleRegressor
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    See `onnx TreeEnsembleRegressor
    <https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html>`_.
    The only change:

    nodes_modes: string contenation with `,`

    **Inputs**

    * X (T1): tensor of type T1

    **Outputs**

    * Y (T2): prediction of type T2

    **Constraints**

    * T1: float, double
    * T2: float, double
    """,
                """
    onnx_extended.ortops.option.cpu.TreeEnsembleRegressorSparse
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It does the sum of two tensors.

    **Provider**

    CPUExecutionProvider

    **Attributes**

    See `onnx TreeEnsembleRegressor
    <https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html>`_.
    The only change:

    nodes_modes: string contenation with `,`

    **Inputs**

    * X (T1): tensor of type T1 (sparse)

    **Outputs**

    * Y (T2): prediction of type T2

    **Constraints**

    * T1: float, double
    * T2: float, double
    """,
            ],
        )
    )
