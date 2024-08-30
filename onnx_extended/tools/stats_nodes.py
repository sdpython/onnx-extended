import pprint
from collections import Counter
from typing import Any, Callable, Dict, Iterator, Iterable, Optional, Tuple, Union
import numpy as np
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
)
from ..reference import CReferenceEvaluator, to_array_extended


def enumerate_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto], recursive: bool = True
) -> Iterable[
    Tuple[
        Tuple[str, ...],
        Union[GraphProto, FunctionProto],
        Union[NodeProto, TensorProto, SparseTensorProto],
    ]
]:
    """
    Enumerates all nodes in a model.

    :param onx: the model
    :param recursive: look into subgraphs
    :return: enumerate tuple *(name, parent, node)*
    """
    if isinstance(onx, ModelProto):
        for c, parent, node in enumerate_nodes(onx.graph, recursive=recursive):
            yield (onx.graph.name, *c), parent, node
        for f in onx.functions:
            for c, parent, node in enumerate_nodes(f, recursive=recursive):
                yield (f.name, *c), parent, node
    elif isinstance(onx, (GraphProto, FunctionProto)):
        if isinstance(onx, GraphProto):
            for init in onx.initializer:
                yield (init.name,), onx, init
            for initp in onx.sparse_initializer:
                yield (initp.indices.name or initp.values.name,), onx, initp
        for i, node in enumerate(onx.node):
            assert isinstance(
                node, NodeProto
            ), f"A NodeProto is expected not {type(node)}."
            if node.op_type == "Constant":
                yield (node.output[0],), onx, node
            else:
                yield (node.name or f"#{i}",), onx, node
            if recursive:
                for att in node.attribute:
                    if att.g:
                        for c, parent, node in enumerate_nodes(
                            att.g, recursive=recursive
                        ):
                            if isinstance(node, NodeProto):
                                n = node.name or f"#{i}"
                            elif isinstance(node, TensorProto):
                                n = node.name
                            elif isinstance(node, SparseTensorProto):
                                n = node.indices.name or node.values.name
                            else:
                                raise TypeError(f"Unexpected type {type(node)}.")
                            yield (f"{n}/{att.name}", *c), parent, node


def extract_attributes(node: NodeProto) -> Dict[str, Tuple[AttributeProto, Any]]:
    """
    Extracts all atributes of a node.

    :param node: node proto
    :return: dictionary
    """
    atts: Dict[str, Tuple[AttributeProto, Any]] = {}
    for att in node.attribute:
        if hasattr(att, "ref_attr_name") and att.ref_attr_name:
            atts[att.name] = (att, None)
            continue
        if att.type == AttributeProto.INT:
            atts[att.name] = (att, att.i)
            continue
        if att.type == AttributeProto.FLOAT:
            atts[att.name] = (att, att.f)
            continue
        if att.type == AttributeProto.INTS:
            atts[att.name] = (att, np.array(att.ints))
            continue
        if att.type == AttributeProto.FLOATS:
            atts[att.name] = (att, np.array(att.floats, dtype=np.float32))
            continue
        if att.type == AttributeProto.GRAPH and hasattr(att, "g") and att.g is not None:
            atts[att.name] = (att, None)
            continue
        if att.type == AttributeProto.SPARSE_TENSORS:
            atts[att.name] = (att, to_array_extended(att.sparse_tensor))
            continue
        if att.type == AttributeProto.TENSOR:
            atts[att.name] = (att, to_array_extended(att.t))
            continue
        if att.type == AttributeProto.TENSORS:
            atts[att.name] = (att, [to_array_extended(t) for t in att.tensors])
            continue
        if att.type == AttributeProto.SPARSE_TENSORS:
            atts[att.name] = (att, [to_array_extended(t) for t in att.sparse_tensors])
            continue
        if att.type == AttributeProto.STRING:
            atts[att.name] = (att, att.s.decode("utf-8"))
            continue
        if att.type == AttributeProto.STRINGS:
            atts[att.name] = (att, np.array([s.decode("utf-8") for s in att.strings]))
            continue
    return atts


class _Statistics:
    """
    Common class to statistics classes.
    """

    def __init__(self):
        self._statistics: Dict[str, Any] = {}

    def __len__(self) -> int:
        "Returns the number of statistics"
        return len(self._statistics)

    def add(self, name: str, value: Any):
        "Adds one statictics."
        assert name not in self._statistics, f"Statistics {name!r} was already added."
        self._statistics[name] = value

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        yield from self._statistics.items()

    def __getitem__(self, name: str) -> Any:
        "Returns one statistics."
        return self._statistics[name]

    def get(self, name: str, default_value: Optional[Any] = None) -> Any:
        "Returns one statistics or a default value if not found."
        return self._statistics.get(name, default_value)

    def __str__(self) -> str:
        "Usual"
        return f"{self.__class__.__name__}(\n{pprint.pformat(self._statistics)})"

    @property
    def dict_values(self) -> Dict[str, Any]:
        """
        Converts the statistics the class holds into a single row in order
        to build a dataframe.
        """
        raise NotImplementedError(
            f"Property 'dict_values' not implemented for class {type(self)}."
        )


class NodeStatistics(_Statistics):
    """
    Stores many statistics for NodeProto.
    """

    def __init__(self, parent: Union[GraphProto, FunctionProto], node: NodeProto):
        _Statistics.__init__(self)
        self.parent = parent
        self.node = node

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(<{self.parent.name}>, <{self.node.op_type}>,\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def dict_values(self) -> Dict[str, Any]:
        "Returns the statistics as a dictionary."
        obs = {}
        for k, v in self._statistics.items():
            if isinstance(
                v, (int, float, str, np.int64, np.int32, np.float32, np.float64)
            ):
                obs[k] = v
            elif isinstance(v, set):
                obs[k] = ",".join(map(str, sorted(v)))
            elif isinstance(v, Counter):
                for kk, vv in v.items():
                    obs[f"{k}__{kk}"] = vv
            elif isinstance(v, list):
                if len(v) == 0:
                    continue
                if isinstance(v[0], (HistTreeStatistics, TreeStatistics)):
                    # It is the statistics for every tree.
                    # Let's skip that.
                    continue
                raise TypeError(
                    f"Unexpected type {type(v)} for statistics {k!r} "
                    f"with element {type(v[0])}."
                )
            elif isinstance(v, _Statistics):
                dv = v.dict_values
                for kk, vv in dv.items():
                    if isinstance(vv, (int, float, str)):
                        obs[f"{k}__{kk}"] = vv
            else:
                raise TypeError(f"Unexpected type {type(v)} for statistics {k!r}: {v}.")
        return obs


class TreeStatistics(_Statistics):
    """
    Stores many statistics on a tree extracted from TreeEnsemble* operators.
    """

    def __init__(self, node: NodeProto, tree_id: int):
        _Statistics.__init__(self)
        self.node = node
        self.tree_id = tree_id

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(<{self.node.op_type}>, {self.tree_id},\n"
            f"{pprint.pformat(self._statistics)})"
        )


class HistTreeStatistics(_Statistics):
    """
    Stores statistics on thresholds.
    """

    def __init__(
        self, node: NodeProto, featureid: int, values: np.ndarray, bins: int = 20
    ):
        _Statistics.__init__(self)
        self.node = node
        self.featureid = featureid
        self.add("min", values.min())
        self.add("max", values.max())
        self.add("mean", values.mean())
        self.add("median", np.median(values))
        self.add("size", len(values))
        n_distinct = len(set(values))
        self.add("n_distinct", n_distinct)
        self.add("hist", np.histogram(values, bins))
        if n_distinct <= 50:
            self.add("v_distinct", set(values))

    def __str__(self) -> str:
        "Usual"
        return (
            f"{self.__class__.__name__}(<{self.node.op_type}>, {self.featureid},\n"
            f"{pprint.pformat(self._statistics)})"
        )


class HistStatistics(_Statistics):
    """
    Stores statistics on constants.
    """

    def __init__(
        self,
        parent: Union[GraphProto, FunctionProto],
        node: Union[NodeProto, TensorProto, SparseTensorProto],
        bins: int = 20,
    ):
        _Statistics.__init__(self)
        self.parent = parent
        self.node = node
        values = self.values

        self.add("sparse", 1 if self.is_sparse else 0)
        self.add("shape", values.shape)
        self.add("dtype", values.dtype)
        self.add("min", values.min())
        self.add("max", values.max())
        self.add("mean", values.mean())
        self.add("median", np.median(values))
        flat = values.ravel()
        self.add("size", values.size)
        n_distinct = len(flat)
        self.add("n_distinct", n_distinct)
        if values.size > 1:
            try:
                self.add("hist", np.histogram(values, bins))
            except IndexError as e:
                raise RuntimeError(
                    f"Unable to process values with shape={values.shape}, "
                    f"dtype={values.dtype}, {values}."
                ) from e
        else:
            self.add("hist", (values, np.array([1], dtype=np.int64)))
        if n_distinct <= 50:
            self.add("v_distinct", set(flat))

    @property
    def dict_values(self) -> Dict[str, Any]:
        "Returns the statistics as a dictionary."
        obs = {}
        for k in [
            "size",
            "shape",
            "dtype",
            "min",
            "max",
            "mean",
            "median",
            "n_distinct",
        ]:
            obs[k] = self[k]
        hist = self["hist"]
        if hist[0].size > 0 and len(hist[0].shape) > 0:
            for i, v in enumerate(hist[0]):
                obs[f"hist_y_{i}"] = v
            for i, v in enumerate(hist[1]):
                obs[f"hist_x_{i}"] = v
        return obs

    @property
    def is_sparse(self) -> bool:
        "Tells if the tensor is sparse."
        return isinstance(self.node, SparseTensorProto)

    @property
    def name(self) -> str:
        "Returns the name of the tensor."
        if isinstance(self.node, SparseTensorProto):
            return self.node.indices.name or self.node.values.name
        if isinstance(self.node, NodeProto):
            return self.node.output[0]
        return self.node.name

    def __str__(self) -> str:
        "Usual"
        if isinstance(self.node, NodeProto):
            return (
                f"{self.__class__.__name__}(<{self.parent.name}>, "
                f"<{self.node.op_type}>,\n"
                f"{pprint.pformat(self._statistics)})"
            )
        return (
            f"{self.__class__.__name__}(<{self.parent.name}>, <{self.name}>,\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def values(self):
        "Returns the values as an array."
        if isinstance(self.node, NodeProto):
            model = CReferenceEvaluator(self.node)
            return model.run(None, {})[0]
        return to_array_extended(self.node)


def stats_tree_ensemble(
    parent: Union[GraphProto, FunctionProto], node: NodeProto
) -> NodeStatistics:
    """
    Computes statistics on every tree of a TreeEnsemble.

    :param parent: function or graph proto hosting the node
    :param node: node
    :return: instance of NodeStatistics
    """
    stats = NodeStatistics(parent, node)
    atts = {k: v[1] for k, v in extract_attributes(node).items()}
    unique = set(atts["nodes_treeids"])
    stats.add("kind", "Regressor" if "n_targets" in atts else "Classifier")
    stats.add("n_trees", len(unique))
    stats.add(
        "n_outputs",
        atts["n_targets"] if "n_targets" in atts else len(atts["class_ids"]),
    )
    stats.add("max_featureid", max(atts["nodes_featureids"]))
    stats.add("n_features", len(set(atts["nodes_featureids"])))
    stats.add("n_rules", len(set(atts["nodes_modes"])))
    stats.add("rules", set(atts["nodes_modes"]))
    stats.add("hist_rules", Counter(atts["nodes_modes"]))

    features = []
    for fid in sorted(set(atts["nodes_featureids"])):
        indices = atts["nodes_featureids"] == fid
        features.append(HistTreeStatistics(node, fid, atts["nodes_values"][indices]))
    stats.add("features", features)

    atts_nodes = {k: v for k, v in atts.items() if k.startswith("nodes")}
    tree_stats = []
    for treeid in sorted(unique):
        tr = TreeStatistics(node, treeid)
        indices = atts_nodes["nodes_treeids"] == treeid
        atts_tree = {k: v[indices] for k, v in atts_nodes.items()}
        tr.add("n_nodes", len(atts_tree["nodes_nodeids"]))
        tr.add("n_leaves", len(atts_tree["nodes_modes"] == "LEAF"))
        tr.add("max_featureid", max(atts_tree["nodes_featureids"]))
        tr.add("n_features", len(set(atts_tree["nodes_featureids"])))
        tr.add("n_rules", len(set(atts_tree["nodes_modes"])))
        tr.add("rules", set(atts_tree["nodes_modes"]))
        tr.add("hist_rules", Counter(atts_tree["nodes_modes"]))
        tree_stats.append(tr)
    stats.add("trees", tree_stats)
    return stats


def stats_constant(
    parent: Union[GraphProto, FunctionProto],
    node: Union[NodeProto, TensorProto, SparseTensorProto],
) -> HistStatistics:
    """
    Computes basic statistics on constants.

    :param parent: function or graph proto hosting the node
    :param node: node
    :return: instance of NodeStatistics
    """
    return HistStatistics(parent, node)


def enumerate_stats_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto],
    recursive: bool = True,
    stats_fcts: Optional[
        Dict[
            Tuple[str, str],
            Callable[
                [
                    Union[GraphProto, FunctionProto],
                    Union[NodeProto, TensorProto, SparseTensorProto],
                ],
                Union[NodeStatistics, HistStatistics],
            ],
        ]
    ] = None,
) -> Iterable[
    Tuple[
        Tuple[str, ...],
        Union[GraphProto, FunctionProto],
        Union[NodeStatistics, HistStatistics],
    ]
]:
    """
    Computes statistics of nodes functions.

    :param onx: the model
    :param recursive: look into subgraphs
    :param stats_fcts: a dicionary of functions to call for every node,
        the key is *(domain, op_type)*, if None, uses the default
        statistiques
    :return: enumerate tuple *(name, parent, statistics)*
    """
    if stats_fcts is None:
        dom_optim = "onnx_extended.ortops.optim.cpu"
        stats_fcts = {
            ("ai.onnx.ml", "TreeEnsembleRegressor"): stats_tree_ensemble,
            ("ai.onnx.ml", "TreeEnsembleClassifier"): stats_tree_ensemble,
            (dom_optim, "TreeEnsembleRegressor"): stats_tree_ensemble,
            (dom_optim, "TreeEnsembleClassifier"): stats_tree_ensemble,
            ("", "Constant"): stats_constant,
        }
    for name, parent, node in enumerate_nodes(onx, recursive=recursive):
        if isinstance(node, NodeProto):
            if (node.domain, node.op_type) in stats_fcts:
                stat = stats_fcts[node.domain, node.op_type](parent, node)
                yield name, parent, stat
        elif ("", "Constant") in stats_fcts:
            stati = stats_fcts["", "Constant"](parent, node)
            if stati["dtype"] in (np.int64, np.int32) and stati["size"] < 10:
                # This is probably a shape. It is skipped.
                continue
            yield name, parent, stati
