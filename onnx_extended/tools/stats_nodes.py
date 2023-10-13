from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from onnx import FunctionProto, GraphProto, ModelProto, NodeProto


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


class NodeStatistics:
    """
    Stores many statistics for NodeProto.
    """

    def __init__(self, parent: Union[GraphProto, FunctionProto], node: NodeProto):
        self.parent = parent
        self.node = node

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(<{self.parent.name}>, <{self.node.op_type}>)"


def stats_tree_ensemble(
    parent: Union[GraphProto, FunctionProto], node: NodeProto
) -> NodeStatistics:
    """
    Computes statistics on every tree of a TreeEnsemble.

    :param parent: function or graph proto hosting the node
    :param node: node
    :return: instance of NodeStatistics
    """
    raise NotImplementedError()


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
