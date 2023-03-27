from typing import Any, Dict
import numpy
from onnx import NodeProto
from onnx.reference.op_run import OpRun
from ._op_classifier_common import _ClassifierCommon
from .c_op_tree_ensemble_p_ import (
    RuntimeTreeEnsembleClassifierPFloat,
    RuntimeTreeEnsembleClassifierPDouble,
)


class TreeEnsembleClassifierCommon(OpRun, _ClassifierCommon):
    op_domain = "ai.onnx.ml"

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(onnx_node, run_params, schema=schema)
        self.parallel = (60, 128, 20)

    def change_parallel(self, trees: int, trees_rows: int, rows: int):
        self.parallel = (trees, trees_rows, rows)
        self._init(dtype=self._dtype, version=self._runtime_version)

    def _init(self, dtype, **kwargs):
        if dtype == numpy.float32:
            cls = RuntimeTreeEnsembleClassifierPFloat
        else:
            cls = RuntimeTreeEnsembleClassifierPDouble

        self.rt_ = cls(self.parallel[0], self.parallel[1], self.parallel[2], True, True)
        self.rt_.init(
            kwargs["aggregate_function"],
            kwargs.get("base_values", []),
            kwargs.get("base_values_as_tensor", []),
            kwargs["n_targets"],
            kwargs["nodes_falsenodeids"],
            kwargs["nodes_featureids"],
            kwargs["nodes_hitrates"],
            kwargs.get("nodes_hitrates_as_tensor", []),
            kwargs.get("nodes_missing_value_tracks_true", []),
            kwargs["nodes_modes"],
            kwargs["nodes_nodeids"],
            kwargs["nodes_treeids"],
            kwargs["nodes_truenodeids"],
            kwargs.get("nodes_values", []),
            kwargs.get("nodes_values_as_tensor", []),
            kwargs["post_transform"],
            kwargs["target_ids"],
            kwargs["target_nodeids"],
            kwargs["target_treeids"],
            kwargs.get("target_weights", []),
            kwargs.get("target_weights_as_tensor", []),
        )

    def _run(self, x, **kwargs):
        """
        This is a C++ implementation coming from
        :epkg:`onnxruntime`.
        `tree_ensemble_classifier.cc
        <https://github.com/microsoft/onnxruntime/blob/master/
        onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc>`_.
        See class :class:`RuntimeTreeEnsembleClassifier
        <mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_.RuntimeTreeEnsembleClassifier>`.
        """
        if hasattr(x, "todense"):
            x = x.todense()
        if self.rt_ is None:
            self.init(**kwargs)
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape((label.shape[0], -1))
        return self._post_process_predicted_label(label, scores)


class TreeEnsembleClassifier_1(TreeEnsembleClassifierCommon):
    def _run(
        self,
        x,
        aggregate_function=None,
        base_values=None,
        class_ids=None,
        class_nodeids=None,
        class_treeids=None,
        class_weights=None,
        classlabels_int64s=None,
        classlabels_strings=None,
        nodes_falsenodeids=None,
        nodes_featureids=None,
        nodes_hitrates=None,
        nodes_missing_value_tracks_true=None,
        nodes_modes=None,
        nodes_nodeids=None,
        nodes_treeids=None,
        nodes_truenodeids=None,
        nodes_values=None,
        post_transform=None,
    ):
        return TreeEnsembleClassifierCommon._run(
            self,
            x,
            base_values=base_values,
            class_ids=None,
            class_nodeids=class_nodeids,
            class_treeids=class_treeids,
            class_weights=class_weights,
            classlabels_int64s=classlabels_int64s,
            classlabels_strings=classlabels_strings,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            post_transform=post_transform,
        )


class TreeEnsembleClassifier_3(TreeEnsembleClassifierCommon):
    def _run(
        self,
        x,
        aggregate_function=None,
        base_values=None,
        base_values_as_tensor=None,
        class_ids=None,
        class_nodeids=None,
        class_treeids=None,
        class_weights=None,
        class_weights_as_tensor=None,
        classlabels_int64s=None,
        classlabels_strings=None,
        nodes_falsenodeids=None,
        nodes_featureids=None,
        nodes_hitrates=None,
        nodes_hitrates_as_tensor=None,
        nodes_missing_value_tracks_true=None,
        nodes_modes=None,
        nodes_nodeids=None,
        nodes_treeids=None,
        nodes_truenodeids=None,
        nodes_values=None,
        nodes_values_as_tensor=None,
        post_transform=None,
    ):
        return TreeEnsembleClassifierCommon._run(
            self,
            x,
            base_values=base_values,
            base_values_as_tensor=base_values_as_tensor,
            class_ids=None,
            class_nodeids=class_nodeids,
            class_treeids=class_treeids,
            class_weights=class_weights,
            class_weights_as_tensor=class_weights_as_tensor,
            classlabels_int64s=classlabels_int64s,
            classlabels_strings=classlabels_strings,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_hitrates_as_tensor=nodes_hitrates_as_tensor,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            nodes_values_as_tensor=nodes_values_as_tensor,
            post_transform=post_transform,
        )
