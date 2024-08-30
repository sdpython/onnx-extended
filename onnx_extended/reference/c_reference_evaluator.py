import os
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.defs import get_schema
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.custom_element_types import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun


def from_array_extended(tensor: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """
    Converts an array into a TensorProto including float 8 types.

    :param tensor: numpy array
    :param name: name
    :return: TensorProto
    """
    dt = tensor.dtype
    if dt == float8e4m3fn and dt.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
        dt_to = np.uint8  # type: ignore[assignment]
    elif dt == float8e4m3fnuz and dt.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
        dt_to = np.uint8  # type: ignore[assignment]
    elif dt == float8e5m2 and dt.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
        dt_to = np.uint8  # type: ignore[assignment]
    elif dt == float8e5m2fnuz and dt.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
        dt_to = np.uint8  # type: ignore[assignment]
    elif dt == bfloat16 and dt.descr[0][0] == "bfloat16":
        to = TensorProto.BFLOAT16
        dt_to = np.uint16  # type: ignore[assignment]
    else:
        return from_array(tensor, name)

    t = from_array(tensor.astype(dt_to), name)
    t.data_type = to
    return t


class CReferenceEvaluator(ReferenceEvaluator):
    """
    This class replaces the python implementation by C implementation
    for a short list of operators quite slow in python (such as `Conv`).
    The class automatically replaces a python implementation
    by a C implementation if available. See example :ref:`l-example-conv`.

    ::

        from onnx.reference import ReferenceEvaluator
        from onnx_extended.reference.c_ops import Conv
        ref = ReferenceEvaluator(..., new_ops=[Conv])

    See :class:`onnx.reference.ReferenceEvaluator` for a detailed documentation.

    **Additions**

    Parameter **save_intermediate** can be set to a folder to save intermediate
    results in this folder. It follows the same design as the backend test.
    Let's consider a model with the following nodes:

    ::

        <
            ir_version: 8,
            opset_import: [ "" : 18]
        >
        agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
        {
            T = MatMul(X, W)
            S = Add(T, B)
            C = Softmax(S)
        }

    It will produce the following files after it is run with
    `CReferenceEvaluator(..., save_intermediate="modelrun")`.

    ::

        modelrun
            +-- test_node_0_MatMul
            |       +-- model.onnx
            |       +-- test_data_set_0
            |               + input_0.pb
            |               + input_1.pb
            |               + output_0.pb
            +-- test_node_1_Add
            |       +-- model.onnx
            |       +-- test_data_set_0
            |               + input_0.pb
            |               + input_1.pb
            |               + output_0.pb
            +-- test_node_2_Softmax
                    +-- model.onnx
                    +-- test_data_set_0
                            + input_0.pb
                            + output_0.pb

    These files can then be run with a different runtime to look for discrepancies.
    Following example executes node by node with onnxruntime.

    ::

        from onnx.backend.test.loader import load_model_tests
        from onnx.reference.c_reference_backend import (
            ReferenceEvaluatorBackend,
            create_reference_backend,
        )
        from onnxruntime import InferenceSession

        root = "folder which folder modelrun"
        examples = load_model_tests(root, "modelrun")

        class Wrapper(InferenceSession):

            def __init__(self, model, *args, providers=None, **kwargs):
                super().__init__(
                    model.SerializeToString(),
                    *args,
                    providers=providers or ["CPUExecutionProvider"],
                    **kwargs,
                )

            def run(self, *args, **kwargs):
                return InferenceSession.run(self, *args, **kwargs)

            @property
            def input_names(self):
                return [i.name for i in self.get_inputs()]

            @property
            def output_names(self):
                return [o.name for o in self.get_outputs()]

        new_cls = ReferenceEvaluatorBackend[NewRef]
        backend = create_reference_backend(new_cls, path_to_test=root)
        beckend.run()

    .. versionadded:: 0.2.0
    """

    @staticmethod
    def default_ops():
        from onnx_extended.reference.c_ops.c_op_conv import Conv
        from onnx_extended.reference.c_ops.c_op_svm_classifier import SVMClassifier
        from onnx_extended.reference.c_ops.c_op_svm_regressor import SVMRegressor
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_regressor import (
            TreeEnsembleRegressor_1,
            TreeEnsembleRegressor_3,
        )
        from onnx_extended.reference.c_ops.c_op_tree_ensemble_classifier import (
            TreeEnsembleClassifier_1,
            TreeEnsembleClassifier_3,
        )
        from onnx_extended.reference.c_custom_ops.custom_op_tree_ensemble_regressor import (  # noqa: E501
            TreeEnsembleRegressor_1 as TreeEnsembleRegressor_1_Float,
            TreeEnsembleRegressor_3 as TreeEnsembleRegressor_3_Float,
        )
        from onnx_extended.reference.c_ops.c_op_tfidf_vectorizer import TfIdfVectorizer
        from onnx_extended.reference.other_ops.op_tokenizer import Tokenizer
        from onnx_extended.reference.other_ops.op_scatternd_of_shape import (
            ScatterNDOfShape,
        )

        return [
            Conv,
            ScatterNDOfShape,
            SVMClassifier,
            SVMRegressor,
            TfIdfVectorizer,
            Tokenizer,
            TreeEnsembleClassifier_1,
            TreeEnsembleClassifier_3,
            TreeEnsembleRegressor_1,
            TreeEnsembleRegressor_3,
            TreeEnsembleRegressor_1_Float,
            TreeEnsembleRegressor_3_Float,
        ]

    @staticmethod
    def filter_ops(
        proto: Union[FunctionProto, ModelProto],
        new_ops: List[OpRun],
        opsets: Dict[str, int],
    ) -> List[OpRun]:
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = {}
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                # not a version
                continue
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed:
                modified.append(cl)
        for k, v in best.items():
            atts = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        new_ops = modified
        return new_ops

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
        save_intermediate: Optional[str] = None,
        **kwargs,
    ):
        if new_ops is None:
            new_ops = CReferenceEvaluator.default_ops()
        else:
            new_ops = new_ops.copy()
            new_ops.extend(CReferenceEvaluator.default_ops())
        new_ops = CReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )

        self.save_intermediate = save_intermediate
        if save_intermediate is not None:
            self._cached_saved_results = {}

    def run(  # type: ignore[override]
        self,
        output_names,
        feed_inputs: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Executes the onnx model.

        :param output_names: requested outputs by names,
            None for all
        :param feed_inputs: dictionary `{ input name: input value }`
        :param attributes: attributes value if the instance runs a FunctionProto
        :return: list of requested outputs
        """
        if output_names is None:
            output_names = self.output_names
        if isinstance(self.proto_, FunctionProto) and attributes is None:
            raise TypeError(f"Unexpected proto type {type(self.proto_)}.")

        # step 1: inputs and initializers
        results = {"": None}
        results.update(self.rt_inits_)
        results.update(feed_inputs)
        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)

        # step 2: execute nodes
        for index, node in enumerate(self.rt_nodes_):
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            inputs = [results[i] for i in node.input]
            linked_attributes = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                if isinstance(value, tuple):
                    raise TypeError(
                        f"Unexected type {type(value)} for output {name!r}."
                    )
                self._log(2, " + %s: %s", name, value)
                results[name] = value
            if self.save_intermediate is not None:
                hidden_names = self._retrieve_hidden_inputs(node.onnx_node)
                set_inputs = set(node.input)
                hidden = {
                    k: v
                    for k, v in results.items()
                    if k in hidden_names and k not in set_inputs
                }
                self._save_intermerdiate_results(index, node, inputs, outputs, hidden)

        # return the results
        list_results: list[Any] = []
        for name in output_names:
            assert name in results, (
                f"Unable to find output name {name!r} "
                f"in {sorted(results)}, proto is\n{self.proto_}"
            )
            list_results.append(results[name])
        return list_results

    @staticmethod
    def _retrieve_input_names(nodes: List[NodeProto]) -> Set[str]:
        inputs = set()
        for node in nodes:
            inputs |= set(node.input)
            for att in node.attribute:
                if att.g:
                    inputs |= CReferenceEvaluator._retrieve_input_names(att.g.node)
        return inputs

    def _retrieve_hidden_inputs(self, node: NodeProto) -> Set[str]:
        names = set()
        for att in node.attribute:
            if att.g:
                names |= CReferenceEvaluator._retrieve_input_names(att.g.node)
        return names

    def _save_intermerdiate_results(
        self,
        index: int,
        node: NodeProto,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
        hidden: Dict[str, np.ndarray],
    ) -> str:
        """
        Saves intermediate results into a folder with
        the same organization as the backend tests.

        :param index: index of the node in the graph
        :param node: node proto
        :param inputs: inputs of the node
        :param outputs: outputs of the node, supposedly the expected outputs
        :param hidden: hidden variables if the node has a subgraph
        :return: test folder
        """

        def get_shape(t):
            return list(t.dims)

        input_protos = [
            from_array_extended(inp, name) for name, inp in zip(node.input, inputs)
        ]
        output_protos = [
            from_array_extended(out, name) for name, out in zip(node.output, outputs)
        ]
        constants = []
        if hidden:
            constants.extend(
                [
                    make_node("Constant", [], [name], value=from_array_extended(value))
                    for name, value in hidden.items()
                ]
            )
        model = make_model(
            make_graph(
                [*constants, node.onnx_node],
                f"test_{node.op_type}",
                [
                    make_tensor_value_info(i.name, i.data_type, get_shape(i))
                    for i in input_protos
                ],
                [
                    make_tensor_value_info(o.name, o.data_type, get_shape(o))
                    for o in output_protos
                ],
            ),
            opset_imports=self.proto_.opset_import,
            ir_version=self.proto_.ir_version,
        )
        model_bytes = model.SerializeToString()
        if model_bytes not in self._cached_saved_results:
            sub = f"test_node_{index}_{node.op_type}"
            path = os.path.join(self.save_intermediate, sub)
            if not os.path.exists(path):
                os.makedirs(path)
            self._cached_saved_results[model_bytes] = [path, -1]
            model_file = os.path.join(path, "model.onnx")
            with open(model_file, "wb") as f:
                f.write(model_bytes)
        path, n_example = self._cached_saved_results[model_bytes]
        self._cached_saved_results[model_bytes][1] += 1
        n_example += 1
        sub_path = os.path.join(path, f"test_data_set_{n_example}")
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        for i, t in enumerate(input_protos):
            with open(os.path.join(sub_path, f"input_{i}.pb"), "wb") as f:
                f.write(t.SerializeToString())
        for i, t in enumerate(output_protos):
            with open(os.path.join(sub_path, f"output_{i}.pb"), "wb") as f:
                f.write(t.SerializeToString())
        return path
