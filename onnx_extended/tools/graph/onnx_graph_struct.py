from typing import List, Union
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

    def __str__(self):
        return f"{self.__class__.__name__}(<{self.proto.__class__.__name__}>)"

    @property
    def op_type(self):
        "Returns the node type."
        return self.proto.op_type

    @property
    def is_node(self):
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
            if i not in self.parent.index_input:
                # An input
                return False
            if not self.is_constant(self.parent.input_index[i]):
                return False
        return True

    @property
    def inputs(self):
        "Input names"
        if self.is_node:
            return self.proto.input
        return []

    @property
    def outputs(self):
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


class NodeSet:
    """
    Defines a set of nodes.
    """

    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
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
            for i in node.inputs:
                self.index_input[i] = node
            for i in node.outputs:
                self.index_output[i] = node

    def __len__(self):
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

    def __iter__(self):
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

        for index in indices:
            if index <= len(self.nodes):
                node = self.nodes[index]
                if node is None:
                    raise RuntimeError(f"Node index {index} was already removed.")
                self.nodes[index] = None
            elif index not in self.nodes_added:
                raise RuntimeError(
                    f"Node index {index} does not exists or was already removed."
                )
            if index in self.removed:
                raise RuntimeError(f"Node index {index} was already removed.")

        for index in indices:
            self.removed.add(index)

        nodes = []
        new_indices = []
        for node in new_nodes:
            n = Node(self.new_index, self, node)
            self.nodes_added[self.new_index] = n
            new_indices.append(self.new_index)
            self.new_index += 1
            nodes.append(n)

        self.nodes_sets[indices[0]] = NodeSet(nodes)
        return new_indices

    def simplify(self):
        """
        Stores every node into nodes.
        """
        self.nodes = list(self)
        self._complete_init()
