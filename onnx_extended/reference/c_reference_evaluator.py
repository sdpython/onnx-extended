from typing import Any, Dict, List, Optional, Union

from onnx import FunctionProto, ModelProto
from onnx.defs import get_schema
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


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

    def default_ops():
        from onnx_extended.reference.c_ops.c_op_conv import Conv
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor import (
            TreeEnsembleRegressor_1,
            TreeEnsembleRegressor_3,
        )
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
            TreeEnsembleClassifier_3,
        )
        from onnx_extended.reference.c_custom_ops.custom_op_tree_ensemble_regressor import (  # noqa: E501
            TreeEnsembleRegressor_1 as TreeEnsembleRegressor_1_Float,
        )

        return [
            Conv,
            TreeEnsembleClassifier_1,
            TreeEnsembleClassifier_3,
            TreeEnsembleRegressor_1,
            TreeEnsembleRegressor_3,
            TreeEnsembleRegressor_1_Float,
        ]

    @staticmethod
    def filter_ops(proto, new_ops, opsets):
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = {}
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                # not a version
                continue
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed:
                modified.append(cl)
        for k, v in best.items():
            atts = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        new_ops = modified
        return new_ops

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
        **kwargs,
    ):
        if new_ops is None:
            new_ops = CReferenceEvaluator.default_ops()
        else:
            new_ops = new_ops.copy()
            new_ops.extend(CReferenceEvaluator.default_ops())
        new_ops = CReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )
