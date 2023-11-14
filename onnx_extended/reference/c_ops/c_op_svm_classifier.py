from typing import Any, Dict
import numpy as np
from onnx import NodeProto
from onnx.reference.op_run import OpRun
from .cpu.c_op_svm_py_ import (
    RuntimeSVMClassifierFloat,
    RuntimeSVMClassifierDouble,
)


class SVMClassifier(OpRun):
    op_domain = "ai.onnx.ml"

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema=schema)
        self.rt_ = None

    def _run(
        self,
        x,
        classlabels_ints=None,
        classlabels_strings=None,
        coefficients=None,
        kernel_params=None,
        kernel_type=None,
        post_transform=None,
        prob_a=None,
        prob_b=None,
        rho=None,
        support_vectors=None,
        vectors_per_class=None,
    ):
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `svm_regressor.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_.
        See class :class:`RuntimeSVMClasssifier
        <mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMClasssifier>`.
        """
        if self.rt_ is None:
            if x.dtype == np.float32:
                self.rt_ = RuntimeSVMClassifierFloat()
            elif x.dtype == np.float64:
                self.rt_ = RuntimeSVMClassifierDouble()
            else:
                raise NotImplementedError(f"Not implemented for dtype={x.dtype}.")
            self.rt_.init(
                classlabels_ints,
                classlabels_strings,
                coefficients,
                kernel_params,
                kernel_type,
                post_transform,
                prob_a,
                prob_b,
                rho,
                support_vectors,
                vectors_per_class,
            )
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape(x.shape[0], pred.shape[0] // x.shape[0])
        if len(pred.shape) == 1:
            pred = pred.reshape((-1, 1))
        return (pred,)
