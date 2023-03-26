from typing import Any, Dict, List, Optional, Union

from onnx import FunctionProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx_extended.reference.c_ops.c_op_conv import Conv
from onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor import (
    TreeEnsembleRegressor_1,
    TreeEnsembleRegressor_3,
)
from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
    TreeEnsembleClassifier_1,
    TreeEnsembleClassifier_3,
)


class CReferenceEvaluator(ReferenceEvaluator):
    """
    This class replaces the python implementation by C implementation
    for a short list of operators quite slow in python (such as `Conv`).
    The class automatically replaces a python implementation
    by a C implementation if available. See example :ref:`l-example-conv`.

    ::

        from onnx.reference import ReferenceEvaluator
        from from onnx.reference.c_ops import Conv
        ref = ReferenceEvaluator(..., new_ops=[Conv])
    """

    default_ops = [
        Conv,
        TreeEnsembleClassifier_1,
        TreeEnsembleClassifier_3,
        TreeEnsembleRegressor_1,
        TreeEnsembleRegressor_3,
    ]

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
    ):
        if new_ops is None:
            new_ops = CReferenceEvaluator.default_ops
        else:
            new_ops = new_ops.copy()
            new_ops.extend(CReferenceEvaluator.default_ops)
        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
        )
