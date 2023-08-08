from typing import Iterable, List, Union
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
)


class Node:
    """
    Defines a node in the graph.
    It can be an iniatialier or a node.
    """

    def __init__(
        self, index: int, parent: "Graph", proto: Union[TensorProto, NodeProto]
    ):
        self.index = index
        self.proto = proto
        self.parent = parent

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.index}, <parent>, <{self.op_type}>) "
            f"[{','.join(self.inputs)}] -> [{','.join(self.outputs)}]"
        )

    @property
    def op_type(self) -> str:
        "Returns the node type."
        return self.proto.op_type

    @property
    def is_node(self) -> bool:
        "True if a NodeProto."
        return isinstance(self.proto, NodeProto)

    def is_constant(self) -> bool:
        """
        True if operator Constant or initializer or a Constant as
        an output of an operator taking only constants.
        """
        if self.is_node:
            if self.proto.op_type == "Constant":
                return True
            return self._is_constant()
        return True

    def _is_constant(self) -> bool:
        "Tells if a node is a constant or operate on constants."
        for i in self.inputs:
            if i not in self.parent.index_output:
                # An input
                return False
            ni = self.parent.index_output[i]
            if not ni.is_constant():
                return False
        return True

    @property
    def inputs(self) -> List[str]:
        "Input names"
        if self.is_node:
            return self.proto.input
        return []

    @property
    def outputs(self) -> List[str]:
        "Output names"
        if self.is_node:
            return self.proto.output
        return [self.proto.name]


class NodeWithSubGraph(Node):
    """
    A node with a subgraphs (If, Loop, Scan, ...).
    """

    def __init__(self, parent: "Graph", proto: NodeProto):
        if not isinstance(proto, NodeProto):
            raise TypeError(f"proto is not a NodeProto but {type(proto)}.")
        super(self).__init__(parent, proto)
        self.subgraphs = {}
        for att in proto.attribute:
            if att.data_type == AttributeProto.GRAPH:
                self.subgraphs[att.name] = Graph(att.g)
        if len(self.subgraphs) == 0:
            raise ValueError(f"A node type {self.proto.op_type!r} has no subgraph.")

    @property
    def inputs(self) -> List[str]:
        raise NotImplementedError(
            f"It should return the implicit inputs for node type {self.op_type!r}."
        )


class NodeSet:
    """
    Defines a set of nodes.
    """

    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterable[Node]:
        for n in self.nodes:
            yield n


class Graph:
    """
    A GraphProto, FunctionProto or ModelProto.
    """

    @staticmethod
    def node_or_node(proto: Union[TensorProto, NodeProto]):
        if isinstance(proto, TensorProto):
            return Node
        for att in proto.attribute:
            if att.type == AttributeProto.GRAPH:
                return NodeWithSubGraph
        return Node

    def __init__(self, proto: Union[FunctionProto, GraphProto, ModelProto]):
        self.proto = proto
        nodes = []
        graph = proto.graph if isinstance(proto, ModelProto) else proto
        if isinstance(graph, GraphProto):
            for init in graph.initializer:
                nodes.append(Node(len(nodes), self, init))
            for init in graph.sparse_initializer:
                nodes.append(Node(len(nodes), self, init))
            self.final_outputs = [o.name for o in graph.output]
        else:
            # FunctionProto
            self.final_outputs = list(graph.output)
        for node in graph.node:
            nodes.append(Graph.node_or_node(node)(len(nodes), self, node))
        self.nodes = nodes
        self._complete_init()

    def _complete_init(self):
        self.removed = set()
        self.index_input = {}
        self.index_output = {}
        self.nodes_added = {}
        self.nodes_sets = {}
        self.new_index = len(self.nodes)

        for node in self.nodes:
            self._complete_init_node(node)

    def _complete_init_node(self, node):
        for i in node.inputs:
            if i not in self.index_input:
                self.index_input[i] = []
            self.index_input[i].append(node)
        for i in node.outputs:
            self.index_output[i] = node

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(...)"

    def __len__(self) -> int:
        "Returns the number of nodes"
        return len(self.nodes) + len(self.nodes_added) - len(self.removed)

    def __getitem__(self, index: int) -> Node:
        """
        Returns a node at a specific index.
        """
        if index < len(self.nodes):
            node = self.nodes[index]
            if node is None:
                if index not in self.nodes_added:
                    raise IndexError(f"Unable to find node index {index}.")
            else:
                return node
        node = self.nodes_added[index]
        if node is None:
            raise IndexError(f"This node was probably reduced {index}.")
        return node

    def __iter__(self) -> Iterable[Node]:
        "Iterates on nodes or initializer."
        for index, node in enumerate(self.nodes):
            if node is None or node.index in self.removed:
                if index in self.nodes_sets:
                    for n in self.nodes_sets[index]:
                        yield n
                continue
            yield node

    def replace(
        self,
        indices: Union[int, List[int]],
        new_nodes: Union[NodeProto, List[NodeProto]],
    ) -> List[int]:
        """
        Replaces a node index

        :param indices: index or list of indices to replace
        :param new_nodes: node or list of nodes to add
        :return: added indices
        """
        if isinstance(new_nodes, NodeProto):
            new_nodes = [new_nodes]
        if isinstance(indices, int):
            indices = [indices]

        removed = []
        for index in indices:
            if index <= len(self.nodes):
                node = self.nodes[index]
                if node is None:
                    raise RuntimeError(f"Node index {index} was already removed.")
                removed.append((index, self.nodes[index]))
                self.nodes[index] = None
            elif index not in self.nodes_added:
                raise RuntimeError(
                    f"Node index {index} does not exists or was already removed."
                )
            if index in self.removed:
                raise RuntimeError(f"Node index {index} was already removed.")

        for index, node in removed:
            self.removed.add(index)
            for i in node.inputs:
                new_input = [n for n in self.index_input[i] if n.index != index]
                self.index_input[i] = new_input
            for o in node.outputs:
                del self.index_output[o]

        nodes = []
        new_indices = []
        for node in new_nodes:
            n = Node(self.new_index, self, node)
            self._complete_init_node(n)
            self.nodes_added[self.new_index] = n
            new_indices.append(self.new_index)
            self.new_index += 1
            nodes.append(n)

        self.nodes_sets[indices[0]] = NodeSet(nodes)
        return new_indices

    def simplify(self, remove_unused: bool = True):
        """
        Stores every node into nodes.
        Removes unused nodes.

        :param remove_unused: removes unused nodes as well,
            see :meth:`remove_unused_nodes`
        """
        if (
            len(self.removed) == 0
            and len(self.nodes_added) == 0
            and len(self.nodes_sets) == 0
        ):
            # Nothing to do.
            return

        self.nodes = list(self)
        self._complete_init()
        for i, node in enumerate(self.nodes):
            node.index = i
        if remove_unused:
            self.remove_unused_nodes()

    def remove_unused_nodes(self):
        """
        Removes unused nodes, a node with only unused outputs.

        :return: removed nodes
        """
        total_remove = []
        while True:
            to_remove = []
            for node in self:
                rem = 0
                for name in node.outputs:
                    if (
                        name not in {"", None}
                        and name not in self.index_input
                        and name not in self.final_outputs
                    ):
                        rem += 1
                if rem < len(node.outputs):
                    # one outputs is used
                    continue
                to_remove.append(node)
                self.removed.add(node.index)

            if len(to_remove) == 0:
                break

            total_remove.extend(to_remove)
            self.simplify(False)
        return total_remove
