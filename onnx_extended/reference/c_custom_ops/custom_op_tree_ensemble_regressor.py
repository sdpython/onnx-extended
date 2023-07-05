from typing import Any, Dict
import numpy
from onnx import NodeProto
from onnx.defs import OpSchema, get_schema
from onnx.reference.op_run import OpRun
from ..c_ops.cpu.c_op_tree_ensemble_py_ import (
    RuntimeTreeEnsembleRegressorFloat,
    RuntimeTreeEnsembleRegressorDouble,
)


class TreeEnsembleRegressorCommon(OpRun):
    op_domain = "onnx_extented.ortops.optim.cpu"

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
            cls = RuntimeTreeEnsembleRegressorFloat
        else:
            cls = RuntimeTreeEnsembleRegressorDouble

        self.rt_ = cls()

        empty_f = numpy.array([], dtype=dtype)
        base_values = numpy.array(
            kwargs.get("base_values", None)
            or kwargs.get("base_values_as_tensor", None)
            or empty_f
        )
        nodes_values = numpy.array(
            kwargs.get("nodes_values", None)
            or kwargs.get("nodes_values_as_tensor", None)
            or empty_f
        )
        nodes_hitrates = numpy.array(
            kwargs.get("nodes_hitrates", None)
            or kwargs.get("nodes_hitrates_as_tensor", None)
            or empty_f
        )
        tw = numpy.array(
            kwargs.get("target_weights", None)
            or kwargs.get("target_weights", None)
            or empty_f
        )

        self.rt_.init(
            kwargs.get("aggregate_function", "SUM"),  # 3
            base_values,  # 4
            kwargs["n_targets"],  # 5
            kwargs["nodes_falsenodeids"],  # 6
            kwargs["nodes_featureids"],  # 7
            nodes_hitrates,  # 8
            kwargs.get("nodes_missing_value_tracks_true", []),  # 9
            kwargs["nodes_modes"].split(","),  # 10
            kwargs["nodes_nodeids"],  # 11
            kwargs["nodes_treeids"],  # 12
            kwargs["nodes_truenodeids"],  # 13
            nodes_values,  # 14
            kwargs["post_transform"],  # 15
            kwargs["target_ids"],  # 16
            kwargs["target_nodeids"],  # 17
            kwargs["target_treeids"],  # 18
            tw,  # 19
        )
        if self.parallel is not None:
            self.rt_.set(*self.parallel)

    def _run(self, x, **kwargs):
        if hasattr(x, "todense"):
            x = x.todense()
        if self.rt_ is None:
            self._init(x.dtype, **kwargs)
        pred = self.rt_.compute(x)
        if pred.shape[0] != x.shape[0]:
            pred = pred.reshape((x.shape[0], -1))
        return (pred,)


def _make_schema():
    attributes = []
    sch = get_schema("TreeEnsembleRegressor", 1, "ai.onnx.ml")
    for att in sch.attributes.values():
        if att.name == "nodes_modes":
            attributes.append(
                OpSchema.Attribute(
                    "nodes_modes",
                    OpSchema.AttrType.STRING,
                    "comma separated value nodes_modes",
                )
            )
        else:
            attributes.append(att)
    return OpSchema(
        "TreeEnsembleRegressor",
        TreeEnsembleRegressorCommon.op_domain,
        1,
        inputs=[
            OpSchema.FormalParameter("X", "T"),
        ],
        outputs=[
            OpSchema.FormalParameter("Y", "T"),
        ],
        type_constraints=[("T", ["tensor(float)"], "")],
        attributes=attributes,
    )


class TreeEnsembleRegressor_1(TreeEnsembleRegressorCommon):
    op_schema = _make_schema()

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
