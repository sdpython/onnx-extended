from typing import Any, Dict
import numpy
from onnx import NodeProto
from onnx.reference.op_run import OpRun
from ._op_classifier_common import _ClassifierCommon
from .cpu.c_op_tree_ensemble_py_ import (
    RuntimeTreeEnsembleClassifierFloat,
    RuntimeTreeEnsembleClassifierDouble,
)


class TreeEnsembleClassifierCommon(OpRun, _ClassifierCommon):
    op_domain = "ai.onnx.ml"

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema=schema)
        self.parallel = None
        self.rt_ = None
        # default is no parallelization
        self.set_parallel(int(100e6), int(100e6), int(100e6), 1, 1, 0)

    def set_parallel(
        self,
        parallel_tree: int = -1,
        parallel_tree_N: int = -1,
        parallel_N: int = -1,
        batch_size_tree: int = -1,
        batch_size_rows: int = -1,
        node3: int = -1,
    ):
        """
        Sets the parameter for parallelization.
        If a parameter is set to -1, its value does not change.

        :param parallel_tree: parallization by trees if the number of trees is higher
        :param parallel_tree_N: batch size (rows) if parallization by trees
        :param parallel_N: parallization by rows if the number of rows is higher
        :param batch_size_tree: number of trees to compute at the same time
        :param batch_size_rows: number of rows to compute at the same time
        :param node3: use bigger nodes
        """
        self.parallel = (
            parallel_tree,
            parallel_tree_N,
            parallel_N,
            batch_size_tree,
            batch_size_rows,
            node3,
        )
        if self.rt_ is not None:
            self.rt_.set(*self.parallel)

    def _init(self, dtype, **kwargs):
        if dtype == numpy.float32:
            cls = RuntimeTreeEnsembleClassifierFloat
        else:
            cls = RuntimeTreeEnsembleClassifierDouble

        empty_f = numpy.array([], dtype=dtype)
        base_values = (
            kwargs.get("base_values", None)
            or kwargs.get("base_values_as_tensor", None)
            or empty_f
        )
        nodes_values = (
            kwargs.get("nodes_values", None)
            or kwargs.get("nodes_values_as_tensor", None)
            or empty_f
        )
        nodes_hitrates = (
            kwargs.get("nodes_hitrates", None)
            or kwargs.get("nodes_hitrates_as_tensor", None)
            or empty_f
        )
        base_values = (
            kwargs.get("base_values", None)
            or kwargs.get("base_values_as_tensor", None)
            or empty_f
        )
        cw = (
            kwargs.get("class_weights", None)
            or kwargs.get("class_weights_as_tensor", None)
            or empty_f
        )
        ncl = max(
            len(kwargs.get("classlabels_int64s", None) or []),
            len(kwargs.get("classlabels_strings", None) or []),
        )
        self.rt_ = cls()
        self.rt_.init(
            "SUM",  # 3
            base_values,  # 4
            ncl,  # 5
            kwargs["nodes_falsenodeids"],  # 6
            kwargs["nodes_featureids"],  # 7
            nodes_hitrates,  # 8
            kwargs["nodes_missing_value_tracks_true"],  # 9
            kwargs["nodes_modes"],  # 10
            kwargs["nodes_nodeids"],  # 11
            kwargs["nodes_treeids"],  # 12
            kwargs["nodes_truenodeids"],  # 13
            nodes_values,  # 14
            kwargs["post_transform"] or "NONE",  # 15
            kwargs["class_ids"],  # 16
            kwargs["class_nodeids"],  # 17
            kwargs["class_treeids"],  # 18
            cw,  # 19
        )
        if self.parallel is not None:
            self.rt_.set(*self.parallel)

    def _run(self, x, **kwargs):
        """
        This is a C++ implementation coming from : epkg:`onnxruntime`.
        `tree_ensemble_classifier.cc <https://github.com/microsoft/onnxruntime/blob/
        master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc>`_.
        """
        if hasattr(x, "todense"):
            x = x.todense()
        if self.rt_ is None:
            self._init(x.dtype, **kwargs)
        label, scores = self.rt_.compute(x)
        if scores.shape[0] != label.shape[0]:
            scores = scores.reshape((label.shape[0], -1))
        cl = kwargs["classlabels_int64s"] or []
        if len(cl) == 0:
            cl = kwargs["classlabels_strings"]
        return self._post_process_predicted_label(label, scores, cl)


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
            aggregate_function=aggregate_function,
            base_values=base_values,
            class_ids=class_ids,
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
            class_ids=class_ids,
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
