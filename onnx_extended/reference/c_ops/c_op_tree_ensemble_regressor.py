from typing import Any, Dict
import numpy
from onnx import NodeProto
from onnx.reference.op_run import OpRun
try:
  from .c_op_tree_ensemble_p_ import (
      RuntimeTreeEnsembleRegressorPFloat,
      RuntimeTreeEnsembleRegressorPDouble,
  )
except ImportError:
  print("Teee")


class TreeEnsembleRegressorCommon(OpRun):
    op_domain = "ai.onnx.ml"

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema=schema)
        self.parallel = (60, 128, 20)

    def change_parallel(self, trees: int, trees_rows: int, rows: int):
        self.parallel = (trees, trees_rows, rows)
        self._init(dtype=self._dtype, version=self._runtime_version)

    def _init(self, dtype, **kwargs):
        if dtype == numpy.float32:
            cls = RuntimeTreeEnsembleRegressorPFloat
        else:
            cls = RuntimeTreeEnsembleRegressorPDouble

            self.rt_ = cls(
                self.parallel[0], self.parallel[1], self.parallel[2], True, True
            )
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
        if hasattr(x, "todense"):
            x = x.todense()
        if self.rt_ is None:
            self.init(**kwargs)
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape((x.shape[0], -1))
        return (pred,)


class TreeEnsembleRegressor_1(TreeEnsembleRegressorCommon):
    def _run(
        self,
        x,
        aggregate_function=None,
        base_values=None,
        n_targets=None,
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
        target_ids=None,
        target_nodeids=None,
        target_treeids=None,
        target_weights=None,
    ):
        return TreeEnsembleRegressorCommon._run(
            self,
            x,
            base_values=base_values,
            n_targets=n_targets,
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
            target_ids=target_ids,
            target_nodeids=target_nodeids,
            target_treeids=target_treeids,
            target_weights=target_weights,
        )


class TreeEnsembleRegressor_3(TreeEnsembleRegressorCommon):
    def _run(
        self,
        x,
        aggregate_function=None,
        base_values=None,
        base_values_as_tensor=None,
        n_targets=None,
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
        target_ids=None,
        target_nodeids=None,
        target_treeids=None,
        target_weights=None,
        target_weights_as_tensor=None,
    ):
        return TreeEnsembleRegressorCommon._run(
            self,
            x,
            base_values=base_values,
            base_values_as_tensor=base_values_as_tensor,
            n_targets=n_targets,
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
            target_ids=target_ids,
            target_nodeids=target_nodeids,
            target_treeids=target_treeids,
            target_weights=target_weights,
            target_weights_as_tensor=target_weights_as_tensor,
        )
