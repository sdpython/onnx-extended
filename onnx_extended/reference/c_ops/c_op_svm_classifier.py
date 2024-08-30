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

    @classmethod
    def _post_process_label_attributes(self, classlabels_int64s, classlabels_strings):
        """
        Replaces string labels by int64 labels.
        It creates attributes *_classlabels_int64s_string*.
        """
        if classlabels_strings:
            class_ints = np.arange(len(classlabels_strings), dtype=np.int64)
            self_classlabels_int64s_string = classlabels_strings
        else:
            class_ints = classlabels_int64s
            self_classlabels_int64s_string = None
        return class_ints, self_classlabels_int64s_string

    @classmethod
    def _post_process_predicted_label(cls, classlabels_int64s_string, label, scores):
        """
        Replaces int64 predicted labels by the corresponding
        strings.
        """
        if classlabels_int64s_string is not None:
            label = np.array([classlabels_int64s_string[i] for i in label])
        return label, scores

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
            (
                classlabels_ints,
                self.classlabels_int64s_string,
            ) = self._post_process_label_attributes(
                classlabels_ints, classlabels_strings
            )
            if x.dtype == np.float32:
                self.rt_ = RuntimeSVMClassifierFloat()
            elif x.dtype == np.float64:
                self.rt_ = RuntimeSVMClassifierDouble()
            else:
                raise NotImplementedError(f"Not implemented for dtype={x.dtype}.")
            self.rt_.init(
                classlabels_ints,
                classlabels_strings.tolist() if classlabels_strings is not None else [],
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
        res = self.rt_.compute(x)
        label, scores = res
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape(label.shape[0], scores.shape[0] // label.shape[0])
        return self._post_process_predicted_label(
            self.classlabels_int64s_string, label, scores
        )
