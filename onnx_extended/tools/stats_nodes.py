import pprint
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto
from onnx.reference.op_run import to_array_extended


def enumerate_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto], recursive: bool = True
) -> Iterable[Tuple[Tuple[str, ...], Union[GraphProto, FunctionProto], NodeProto]]:
    """
    Enumerates all nodes in a model.

    :param onx: the model
    :param recursive: look into subgraphs
    :return: enumerate tuple *(name, parent, node)*
    """
    if isinstance(onx, ModelProto):
        for c, parent, node in enumerate_nodes(onx.graph, recursive=recursive):
            yield (onx.graph.name,) + c, parent, node
        for f in onx.functions:
            for c, parent, node in enumerate_nodes(f, recursive=recursive):
                yield (f.name,) + c, parent, node
    elif isinstance(onx, (GraphProto, FunctionProto)):
        for i, node in enumerate(onx.node):
            yield (node.name or f"#{i}",), onx, node
            if recursive:
                for att in node.attribute:
                    if att.g:
                        for c, parent, node in enumerate_nodes(
                            att.g, recursive=recursive
                        ):
                            n = node.name or f"#{i}"
                            yield (f"{n}/{att.name}",) + c, parent, node


def extract_attributes(node: NodeProto) -> Dict[str, Tuple[AttributeProto, Any]]:
    """
    Extracts all atributes of a node.

    :param node: node proto
    :return: dictionary
    """
    atts = {}
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
            atts[att.name] = (att, to_array_extended(att.t))
            continue
        if att.type == AttributeProto.TENSOR:
            atts[att.name] = (att, to_array_extended(att.sp))
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
        if name in self._statistics:
            raise ValueError(f"Statistics {name!r} was already added.")
        self._statistics[name] = value

    def __iter__(self) -> Iterable[Tuple[str, Any]]:
        for it in self._statistics.items():
            yield it

    def __getitem__(self, name: str) -> Any:
        "Returns one statistics."
        return self._statistics[name]

    def __str__(self):
        "Usual"
        return f"{self.__class__.__name__}(\n{pprint.pformat(self._statistics)})"


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


class HistStatistics(_Statistics):
    """
    Stores statistics on thresholds.
    """

    def __init__(self, node, featureid, values, bins=20):
        _Statistics.__init__(self)
        self.node = node
        self.featureid = featureid
        self.add("min", values.min())
        self.add("max", values.max())
        self.add("mean", values.mean())
        self.add("median", np.median(values))
        self.add("n", len(values))
        n_distinct = len(set(values))
        self.add("n_distinct", n_distinct)
        self.add("hist", np.histogram(values, bins))
        if n_distinct <= 50:
            self.add("v_distinct", set(values))

    def __str__(self):
        "Usual"
        return (
            f"{self.__class__.__name__}(<{self.node.op_type}>, {self.featureid},\n"
            f"{pprint.pformat(self._statistics)})"
        )


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

    features = []
    for fid in sorted(set(atts["nodes_featureids"])):
        indices = atts["nodes_featureids"] == fid
        features.append(HistStatistics(node, fid, atts["nodes_values"][indices]))
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
        tree_stats.append(tr)
    stats.add("trees", tree_stats)

    return stats


def enumerate_stats_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto],
    recursive: bool = True,
    stats_fcts: Optional[
        Dict[
            Tuple[str, str],
            Callable[[Union[GraphProto, FunctionProto], NodeProto], NodeStatistics],
        ]
    ] = None,
) -> Iterable[Tuple[Tuple[str, ...], Union[GraphProto, FunctionProto], NodeStatistics]]:
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
        dom_optim = "onnx_extented.ortops.optim.cpu"
        stats_fcts = {
            ("ai.onnx.ml", "TreeEnsembleRegressor"): stats_tree_ensemble,
            ("ai.onnx.ml", "TreeEnsembleClassifier"): stats_tree_ensemble,
            (dom_optim, "TreeEnsembleRegressor"): stats_tree_ensemble,
            (dom_optim, "TreeEnsembleClassifier"): stats_tree_ensemble,
        }
    for name, parent, node in enumerate_nodes(onx, recursive=recursive):
        if (node.domain, node.op_type) in stats_fcts:
            stat = stats_fcts[node.domain, node.op_type](parent, node)
            yield name, parent, stat
