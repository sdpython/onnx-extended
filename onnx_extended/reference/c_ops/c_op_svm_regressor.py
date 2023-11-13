from typing import Any, Dict
import numpy as np
from onnx import NodeProto
from onnx.reference.op_run import OpRun
from .cpu.c_op_svm_py_ import (
    RuntimeSVMRegressorFloat,
    RuntimeSVMRegressorDouble,
)


class SVMRegressor(OpRun):
    op_domain = "ai.onnx.ml"

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema=schema)
        self.rt_ = None

    def _run(
        self,
        x,
        coefficients=None,
        kernel_params=None,
        kernel_type=None,
        n_supports=None,
        one_class=None,
        post_transform=None,
        rho=None,
        support_vectors=None,
    ):
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `svm_regressor.cc
        <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_.
        See class :class:`RuntimeSVMRegressor
        <mlprodict.onnxrt.ops_cpu.op_svm_regressor_.RuntimeSVMRegressor>`.
        """
        if self.rt_ is None:
            if x.dtype == np.float32:
                self.rt_ = RuntimeSVMRegressorFloat()
            elif x.dtype == np.float64:
                self.rt_ = RuntimeSVMRegressorDouble()
            else:
                raise NotImplementedError(f"Not implemented for dtype={x.dtype}.")
            self.rt_.init(
                coefficients,
                kernel_params,
                kernel_type,
                n_supports,
                one_class,
                post_transform,
                rho,
                support_vectors,
            )
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape(x.shape[0], pred.shape[0] // x.shape[0])
        if len(pred.shape) == 1:
            pred = pred.reshape((-1, 1))
        return (pred,)
