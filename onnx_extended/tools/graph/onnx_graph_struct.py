from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    ValueInfoProto,
)
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    set_model_props,
)
from onnx.version_converter import convert_version
from ...reference import CReferenceEvaluator
from ...reference.c_reference_evaluator import from_array_extended


class Node:
    """
    Defines a node in the graph.
    It can be an iniatialier or a node.
    """

    def __init__(
        self,
        index: int,
        parent: "Graph",
        proto: Union[TensorProto, NodeProto, ValueInfoProto, str],
    ):
        if not isinstance(proto, (TensorProto, NodeProto, ValueInfoProto, str)):
            raise TypeError(f"Unexpected type {type(proto)} for proto.")
        if isinstance(proto, NodeProto) and proto.op_type == "Constant":
            missing = True
            for att in proto.attribute:
                if att.name in {
                    "sparse_value",
                    "value",
                    "value_float",
                    "value_floats",
                    "value_int",
                    "value_ints",
                    "value_string",
                    "value_strings",
                }:
                    missing = False
                    break
            if missing:
                raise ValueError(f"Unexpected constant node {proto}.")
        self.index = index
        self.proto = proto
        self.parent = parent

    def get_tensor(self) -> TensorProto:
        "Returns the value of the"
        if self.is_node:
            if self.op_type == "Constant":
                model = CReferenceEvaluator(self.proto)
                arr = model.run(None, {})[0]
                return from_array_extended(arr, name=self.outname)
            raise NotImplementedError(
                f"{self.outname!r} is a constant obtained from other constant. "
                f"This case is not implemented yet."
            )
        if self.is_input:
            raise RuntimeError(f"{self.outname!r} is an input not a tensor.")
        return self.proto

    @property
    def outname(self):
        "Returns the output name."
        if len(self.outputs) != 1:
            raise RuntimeError(f"The node has more than one output: {self.outputs}.")
        return self.outputs[0]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.index}, <parent>, <{self.op_type}>) "
            f"[{','.join(self.inputs)}] -> [{','.join(self.outputs)}]"
        )

    @property
    def is_node(self) -> bool:
        "True if a NodeProto."
        return isinstance(self.proto, NodeProto)

    @property
    def is_input(self) -> bool:
        "True if an input"
        return isinstance(self.proto, (ValueInfoProto, str))

    @property
    def is_initializer(self) -> bool:
        "True if inititializer"
        return isinstance(self.proto, TensorProto)

    @property
    def is_sparse_initializer(self) -> bool:
        "True if inititializer"
        return isinstance(self.proto, SparseTensorProto)

    @property
    def op_type(self) -> str:
        "Returns the node type."
        if self.is_input:
            # It is an input.
            return "input"
        return self.proto.op_type if self.is_node else "initializer"

    def is_constant(self) -> bool:
        """
        True if operator Constant or initializer or a Constant as
        an output of an operator taking only constants.
        """
        if self.is_node:
            if self.proto.op_type == "Constant":
                return True
            return self._is_constant()
        return not self.is_input

    def _is_constant(self) -> bool:
        "Tells if a node is a constant or operate on constants."
        # This function is recursive and its results may
        # be cached for better performance.
        for i in self.inputs:
            if i not in self.parent.index_output:
                raise RuntimeError(f"Unable to find output {i!r} in the graph.")
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

    def create_with_new_values(self, new_tensor: TensorProto) -> "Node":
        "Creates an iniatializer or a node Constant based on the new value."
        if self.is_node:
            new_name = self.parent.generate_name(new_tensor.name)
            node = make_node("Constant", [], [new_name], value=new_tensor)
            return Node(None, self.parent, node)
        # initializer
        new_tensor.name = self.parent.generate_name(new_tensor.name)
        return Node(None, self.parent, new_tensor)


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
    def node_or_node(proto: Union[TensorProto, NodeProto, ValueInfoProto, str]):
        if isinstance(proto, (TensorProto, ValueInfoProto, str)):
            return Node
        for att in proto.attribute:
            if att.type == AttributeProto.GRAPH:
                return NodeWithSubGraph
        return Node

    def _get_nodes(
        self, graph: Union[GraphProto, FunctionProto]
    ) -> Tuple[List[Node], List[str]]:
        """
        Returns the ordered list of nodes.
        """
        nodes = []
        if isinstance(graph, GraphProto):
            for inp in graph.input:
                nodes.append(Node(len(nodes), self, inp))
            for init in graph.initializer:
                nodes.append(Node(len(nodes), self, init))
            for init in graph.sparse_initializer:
                nodes.append(Node(len(nodes), self, init))
            graph_inputs = [o.name for o in graph.input]
            graph_outputs = [o.name for o in graph.output]
        else:
            for inp in graph.input:
                nodes.append(Node(len(nodes), self, inp))
            graph_inputs = [o.name for o in graph.input]
            graph_outputs = [o.name for o in graph.output]
        for node in graph.node:
            nodes.append(Graph.node_or_node(node)(len(nodes), self, node))
        return nodes, graph_inputs, graph_outputs

    def __init__(self, proto: Union[FunctionProto, GraphProto, ModelProto]):
        self.proto = proto
        self.functions: Optional[Dict[self, FunctionProto]] = None
        if isinstance(proto, ModelProto):
            graph = proto.graph
            if len(proto.functions) > 0:
                raise NotImplementedError(
                    "Class Graph does not handle model included functions yet."
                )
            self.functions = {f.name: f for f in proto.functions}
        else:
            graph = proto
        self.nodes, self.graph_inputs, self.graph_outputs = self._get_nodes(graph)
        self.opsets: Dict[str, int] = {}
        self.functions: Dict[Tuple[str, str], FunctionProto] = {}
        self._complete_init()

    def _complete_init(self):
        self.removed: Set[str] = set()
        self.index_input: Dict[str, List[Node]] = {}
        self.index_output: Dict[str, Node] = {}
        self.nodes_added: Dict[int, Node] = {}
        self.nodes_sets: Dict[int:NodeSet] = {}
        self.generated_names: Set[str] = set()
        self.new_index: int = len(self.nodes)

        for node in self.nodes:
            self._complete_init_node(node)

    def _complete_init_node(self, node):
        for i in node.inputs:
            if i not in self.index_input:
                self.index_input[i] = []
            self.index_input[i].append(node)
        for i in node.outputs:
            self.index_output[i] = node

    def _exists_name(self, name):
        if name in self.index_input:
            return True
        if name in self.index_output:
            return True
        if name in self.graph_inputs:
            return True
        if name in self.generated_names:
            return True
        return False

    def generate_name(self, prefix: str = "new") -> str:
        """
        Generates a name which is not used for any existing result in the graph.

        :param prefix: prefix to use for the new name,
            next tries will be ``<prefix>_1``, ``<prefix>_2``, ...
        :return: new name
        """
        suggestion = prefix
        i = 0
        while self._exists_name(suggestion):
            i += 1
            suggestion = f"{prefix}_{i}"
        self.generated_names.add(suggestion)
        return suggestion

    def get_node_producer(self, name: str) -> Node:
        """
        Returns the node producing the output *name*.

        :param name: output name to check
        :return: Node producing the output *name* or None if it is an input.
        """
        if name not in self.index_input:
            raise ValueError(f"Unable to find any node producing output {name!r}.")
        return self.index_output[name]

    def get_opsets(self) -> Dict[str, int]:
        """
        Returns the opsets available registered for ever domain
        in the model.
        """
        if not isinstance(self.proto, ModelProto):
            raise TypeError(
                f"The graph does not represent a ModelProto but {type(self.proto)}."
            )
        res = {}
        for op in self.proto.opset_import:
            res[op.domain] = op.version
        res.update(self.opsets)
        return res

    def get_opset(self, domain: str = "") -> int:
        """
        Returns the opset for a specific domain.

        :param domain: domain
        :return: model opset
        """
        if not isinstance(self.proto, ModelProto):
            raise TypeError(
                f"The graph does not represent a ModelProto but {type(self.proto)}."
            )
        for op in self.proto.opset_import:
            if op.domain == domain:
                return op.version
        if domain in self.opsets:
            return self.opsets[domain]
        raise RuntimeError(f"Domain {domain!r} is not part the the model.")

    def is_constant(self, name: str) -> bool:
        """
        Tells if output *name* is constant.

        :param name: result name
        :return: True if constant
        """
        if name in self.graph_inputs:
            return False
        node = self.index_output[name]
        return node.is_constant()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(...) "
            f"[{','.join(self.graph_inputs)}] -> [{','.join(self.graph_outputs)}]"
        )

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

    def replace_nodes(
        self,
        indices: Union[int, List[int]],
        new_nodes: Union[NodeProto, List[NodeProto]],
        new_opsets: Optional[Dict[str, int]] = None,
    ) -> NodeSet:
        """
        Replaces a node index

        :param indices: index or list of indices to replace
        :param new_nodes: node or list of nodes to add
        :param new_opsets: new opet versions
        :return: added nodes

        By default, the nodes are inserted at position `indices[-1]`.
        It ensures the inputs of the new nodes were already computed.
        However, it does not ensure that every intermediate node
        between the first and the last removed nodes can be computed.
        Sorting the nodes is needed in that. This function does not do
        that.
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

        nodes_set = NodeSet(nodes)
        new_pos = indices[-1]
        if new_pos in self.nodes_sets:
            raise NotImplementedError(
                f"Nodes were already added at position {new_pos}. "
                f"This conflicts is not yet handled."
            )
        self.nodes_sets[new_pos] = nodes_set
        if new_opsets is not None:
            self.opsets.update(new_opsets)
        return nodes_set

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
                        and name not in self.graph_outputs
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

    def upgrade_opsets(self, new_opsets: Dict[str, int]):
        """
        Upgrades the models to a newer opsets.

        :param new_opsets: dictionary { domain: new version }
        """
        if not isinstance(self.proto, ModelProto):
            raise RuntimeError(
                f"Upgrading a model only works on a ModelProto not {type(self.proto)}."
            )
        if len(new_opsets) != 1 or "" not in new_opsets:
            raise RuntimeError(
                f"Upgrade an opset only work for domain '' "
                f"but new_opsets={new_opsets}."
            )
        new_proto = convert_version(self.proto, new_opsets[""])
        self.proto = new_proto
        self.nodes, self.graph_inputs, self.graph_outputs = self._get_nodes(
            self.proto.graph
        )
        self._complete_init()

    def add_functions(self, protos: Iterable[FunctionProto]):
        """
        Adds functions to the graph when it is exported to ONNX.

        :param protos: enumerate of FunctionProto
        """
        for proto in protos:
            if not isinstance(proto, FunctionProto):
                raise TypeError(f"Unexpected type {type(proto)} for a function.")
            key = proto.domain, proto.name
            if key in self.functions:
                raise ValueError(
                    f"Function {proto.name!r} from domain "
                    f"{proto.domain!r} as already added."
                )
            self.functions[key] = proto

    def to_onnx(self) -> Union[ModelProto, FunctionProto, GraphProto]:
        """
        Converts the current graph into onnx with the same type
        as the input type.
        """
        if isinstance(self.proto, ModelProto):
            opsets = self.get_opsets()
            initializer = [n.proto for n in self if n.is_initializer]
            sparse_initializer = [n.proto for n in self if n.is_sparse_initializer]
            nodes = [n.proto for n in self if n.is_node]
            model = make_model(
                make_graph(
                    nodes,
                    self.proto.graph.name,
                    self.proto.graph.input,
                    self.proto.graph.output,
                    initializer=initializer,
                    sparse_initializer=sparse_initializer,
                ),
                ir_version=self.proto.ir_version,
                producer_name=self.proto.producer_name,
                producer_version=self.proto.producer_version,
                domain=self.proto.domain,
                model_version=self.proto.model_version,
                doc_string=self.proto.doc_string,
                # training_info=self.proto.training_info,
                opset_imports=[make_opsetid(k, v) for k, v in opsets.items()],
                functions=None
                if len(self.functions) == 0
                else list(self.functions.values()),
            )
            if len(self.proto.metadata_props) > 0:
                set_model_props(
                    model, {p.key: p.value for p in self.proto.metadata_props}
                )
            return model

        raise NotImplementedError(
            f"The conversion to onnx is not implemented for type {type(self.proto)}."
        )
