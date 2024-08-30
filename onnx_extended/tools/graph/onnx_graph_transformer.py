from enum import IntEnum
from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import FunctionProto, NodeProto, TensorProto
from onnx.helper import (
    make_node,
    make_tensor,
    make_tensor_value_info,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.custom_element_types import float8e4m3fn
from onnx.reference.op_run import to_array_extended
from onnx.onnx_cpp2py_export.defs import SchemaError

try:
    from onnx.reference.ops.op_cast import Cast_19 as Cast
    from onnx.reference.ops.op_quantize_linear import (
        QuantizeLinear_19 as QuantizeLinear,
    )
except ImportError:
    from onnx.reference.ops.op_cast import Cast
    from onnx.reference.ops.op_quantize_linear import QuantizeLinear
from ...helper import (
    make_dynamic_quantize_linear_function_proto,
    make_matmul_reshape_transpose_back_function_proto,
    make_matmul_reshape_transpose_function_proto,
    make_simple_dynamic_quantize_linear_function_proto,
)
from ...reference import from_array_extended
from ...validation.cython.fp8 import cast_float32_to_e4m3fn
from .errors import QuantizationError
from .onnx_graph_struct import _get_shape, Graph, Node, NodeKind


logger = getLogger("onnx-extended/transformer")


class QuantizeOptions(IntEnum):
    """
    Quantization options.

    * `NONE`: no option
    * `OPTIMIZE`: assumes there is no nan values,
        choose less generic functions such as the implementation
        of DynamicQuantizeLinear
    """

    NONE = 0
    OPTIMIZE = 1


class TransformResults:
    """
    Output of a function transforming a graph.

    :param removed_nodes: node to remove from the graph
    :param added_nodes: node to add to the graph
    :param new_opsets: opsets to update
    :param local_functions: necessary functions to add to the graph
    """

    def __init__(
        self,
        removed_nodes: List[Node],
        added_nodes: List[NodeProto],
        new_opsets: Optional[Dict[str, int]] = None,
        local_functions: Optional[List[FunctionProto]] = None,
    ):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.new_opsets = new_opsets or {}
        self.local_functions = local_functions or []


def estimation_quantization_scale(
    coef: np.array,
    to: int = TensorProto.FLOAT8E4M3FN,
    threshold: float = 0.99999,
) -> Tuple[float, float]:
    """
    Estimates the scale parameter for the quantization to float 8 assuming
    the distribution of the coefficients is gaussian.
    """

    if to in (
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    ):
        if to == TensorProto.FLOAT8E4M3FN:
            fct = float8e4m3_to_float32
        elif to == TensorProto.FLOAT8E4M3FNUZ:

            def fct(x):
                return float8e4m3_to_float32(x, uz=True)

        elif to == TensorProto.FLOAT8E5M2:
            fct = float8e5m2_to_float32
        elif to == TensorProto.FLOAT8E5M2FNUZ:

            def fct(x):
                return float8e5m2_to_float32(x, uz=True, fn=True)

        else:
            raise ValueError(f"Unexpected to={to!r}.")

        float8 = [fct(i) for i in range(256)]
        quant_float = [f for f in float8 if not np.isnan(f) and not np.isinf(f)]
        std_coef = np.mean(coef.ravel() ** 2) ** 0.5
        std_quant = np.std(np.array(quant_float, dtype=np.float32))
        zero = 0.0
        scale = std_quant.astype(coef.dtype) / std_coef.astype(coef.dtype)
    elif to == TensorProto.UINT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 255 / (qu[1] - qu[0])
        zero = qu[0] * scale
    elif to == TensorProto.INT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 254 / (qu[1] - qu[0])
        zero = (qu[0] + qu[1]) / 2 * scale
    else:
        raise ValueError(f"Unexpected quantization type for to={to}.")

    return np.array(1.0 / scale, dtype=coef.dtype), -zero


def quantize_weights(
    node: Node,
    elem_type: int,
    index: int,
    transpose: bool = False,
) -> Tuple[Node, Node, Node, Union[bool, Tuple[int, ...]], Tuple[int, ...]]:
    """
    Quantizes a tensor into a tensor of element type *elem_type*.

    :param node: Node to quantize
    :param elem_type: element type
    :param transpose: transpose the weight before doing it
    :param verbose: logs what is quantized
    :return: three new nodes, quantized weights, scale, zero point
        and the original shape of the constant, shape

    See function :func:`quantize_float8_matmul`
    """
    tensor = node.get_tensor()
    values = to_array_extended(tensor)
    shape = values.shape
    if len(values.shape) == 1:
        raise NotImplementedError(
            f"Input {index} is a constant, it must have at "
            f"least 2 dimensions not {values.shape}."
        )
    if len(values.shape) > 2:
        # Needs to be reshaped.
        reshaped = values.shape
        if index == 0:
            # first input
            values = values.reshape((-1, values.shape[-1]))
        else:
            # second input
            values = (
                values.reshape((-1,) + values.shape[-2:])
                .transpose((1, 0, 2))
                .reshape((values.shape[-2], -1))
            )
    else:
        reshaped = False
    if transpose:
        values = values.T
    scale, zp = estimation_quantization_scale(values, to=elem_type)
    zpt = make_tensor("zp", elem_type, [], [zp])
    zpa = to_array_extended(zpt)

    if elem_type == TensorProto.FLOAT8E4M3FN:
        new_values = cast_float32_to_e4m3fn(values / scale).astype(float8e4m3fn)
    else:
        new_values = QuantizeLinear.eval(values, scale, zpa)

    new_tensor = from_array_extended(
        new_values, name=node.parent.generate_name(node.outname)
    )
    node_weight = node.create_with_new_values(new_tensor)

    new_scale = from_array_extended(
        scale, name=node.parent.generate_name(f"{node.outname}_scale")
    )
    node_scale = node.create_with_new_values(new_scale)

    new_zp = from_array_extended(
        zpa, name=node.parent.generate_name(f"{node.outname}_zp")
    )
    node_zp = node.create_with_new_values(new_zp)

    return node_weight, node_scale, node_zp, reshaped, shape


class _MainQuantizeState:
    def __init__(self, version, domain_ops, local_functions, opset, index_transpose):
        self.domain_dq = "local.quant.domain"
        if domain_ops is None:
            domain_ops = {}
        self.domain_ops = domain_ops
        if version == "onnxruntime":
            self.op_gemm = "GemmFloat8"
            self.domain_gemm = domain_ops.get("GemmFloat8", "com.microsoft")
        elif version == "onnx-extended":
            self.op_gemm = "CustomGemmFloat8E4M3FN"
            self.domain_gemm = domain_ops.get(
                "CustomGemmFloat8E4M3FN", "onnx_extended.ortops.tutorial.cuda"
            )
        else:
            raise ValueError(f"Unexpected value {version!r} for version.")
        if local_functions is None:
            raise ValueError("local_functions cannot be None")
        self.local_functions = local_functions
        self.opset = opset
        self.index_transpose = index_transpose


class _QuantizeState:
    def __init__(self, logger, quantize_options, main_state):
        self.removed = []
        self.added = []
        self.input_names = []
        self.was_reshaped = [False, False]
        self.shapes = []
        self.logger = logger
        self.quantize_options = quantize_options
        self.main_state = main_state

    def quantize_connstant_weights(self, node, name, elem_type, index, do_transpose):
        cst = node.parent.get_node_producer(name)
        weight, scale, zero_point, reshaped, shape = quantize_weights(
            cst, elem_type, index, transpose=do_transpose
        )
        self.added.extend([weight.proto, scale.proto, zero_point.proto])
        self.input_names.append([weight.outname, scale.outname, zero_point.outname])
        self.removed.append(cst)
        self.was_reshaped[index] = reshaped
        self.shapes.append(shape)
        self.logger.debug("[quantize_weights] static quantize shape=%r", shape)

    def quantize_dynamic(self, node, name, elem_type, index, do_transpose):
        # Not a constant
        shape = node.parent.get_shape(name)
        self.shapes.append(shape)
        self.logger.debug("[quantize_float8_matmul] quantize dynamic shape %r", shape)
        if len(shape) == 2:
            if do_transpose:
                # no need
                temp_name = node.parent.generate_name(f"{name}_tr")
                self.added.append(
                    make_node(
                        "Transpose",
                        [name],
                        [temp_name],
                        perm=[1, 0],
                        name=node.parent.generate_node_name(f"tra8_{name}"),
                    )
                )
            else:
                temp_name = name
        else:
            if shape is None:
                raise QuantizationError(
                    f"Shape is unknown for result {name!r} in node {node}. "
                    f"This input cannot be transposed with certainty."
                )

            temp_name = node.parent.generate_name(f"{name}_tr")
            fname = f"MatMulReshapeTranspose{'T' if do_transpose else 'N'}{index}"
            self.added.append(
                make_node(
                    fname,
                    [name],
                    [temp_name],
                    perm=[1, 0],
                    domain=self.main_state.domain_dq,
                    name=node.parent.generate_node_name("MMRT"),
                )
            )
            self.was_reshaped[index] = name
            if (
                self.main_state.domain_dq,
                fname,
            ) not in self.main_state.local_functions:
                # use local functions
                self.main_state.local_functions[self.main_state.domain_dq, fname] = (
                    make_matmul_reshape_transpose_function_proto(
                        domain=self.main_state.domain_dq,
                        opset=self.main_state.opset,
                        index=index,
                        transpose=do_transpose,
                    )
                )

        new_name = node.parent.generate_name(f"{name}_f8")
        scale = node.parent.generate_name(f"{name}_scale")
        zero_point = node.parent.generate_name(f"{name}_zp")

        if self.quantize_options == QuantizeOptions.NONE:
            fname, fmake = (
                "DynamicQuantizeLinear",
                make_dynamic_quantize_linear_function_proto,
            )
        elif self.quantize_options == QuantizeOptions.OPTIMIZE:
            suffix = {
                TensorProto.FLOAT8E4M3FN: "E4M3FN",
                TensorProto.FLOAT8E4M3FNUZ: "E4M3FNUZ",
                TensorProto.FLOAT8E5M2: "E5M2",
                TensorProto.FLOAT8E5M2FNUZ: "E5M2FNUZ",
            }
            fname, fmake = (
                f"DynamicQuantizeLinear{suffix[elem_type]}",
                lambda domain=None, opset=None: (
                    make_simple_dynamic_quantize_linear_function_proto(
                        domain=domain, opset=opset, to=elem_type
                    )
                ),
            )
        else:
            raise RuntimeError(
                f"Unexpected value {self.quantize_options!r} for quantize_options."
            )
        proto = make_node(
            fname,
            [temp_name],
            [new_name, scale, zero_point],
            to=elem_type,
            domain=self.main_state.domain_dq,
            name=node.parent.generate_node_name("DQL"),
        )
        dql = Node(None, node.parent, proto, NodeKind.NODE)
        self.added.extend([dql.proto])
        self.input_names.append(dql.outputs)
        if (
            self.main_state.domain_dq == "local.quant.domain"
            and (self.main_state.domain_dq, fname)
            not in self.main_state.local_functions
        ):
            # use local functions
            self.main_state.local_functions[self.main_state.domain_dq, fname] = fmake(
                domain=self.main_state.domain_dq, opset=self.main_state.opset
            )

    def finalize(self, node, name, output_type):
        if self.was_reshaped[0] and self.was_reshaped[1]:
            raise QuantizationError(
                f"MatMul cannot be replaced by operator Gemm as both inputs "
                f"are not matrices. Their shapes are {self.shapes}. "
                f"Node name is {node.name!r}."
            )

        if output_type in {
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }:
            # output is quantized, there is a need for a scale
            scale_out = node.parent.generate_name(f"{name}_scaleout")
            self.added.append(
                make_node(
                    "Mul",
                    [self.input_names[0][1], self.input_names[1][1]],
                    [scale_out],
                    name=node.parent.generate_node_name(f"mul8_{name}"),
                )
            )
        else:
            # output is not quantized, no need for an output scale
            scale_out = ""

        gemm_inputs = [
            self.input_names[0][0],  # A
            self.input_names[1][0],  # B
            "",  # C
            self.input_names[0][1],  # scaleA
            self.input_names[1][1],  # scaleB
            scale_out,  # scaleR
        ]
        if self.was_reshaped[0] or self.was_reshaped[1]:
            gemm_outputs = [node.parent.generate_name(f"{name}_gemm")]
            do_reshape = True
        else:
            gemm_outputs = node.outputs
            do_reshape = False

        atts = dict(
            dtype=output_type,
            transA=1 if (self.main_state.index_transpose & 1) else 0,
            transB=1 if (self.main_state.index_transpose & 2) else 0,
            domain=self.main_state.domain_gemm,
            name=node.parent.generate_node_name("GEMMFP8"),
        )
        if self.main_state.domain_gemm != "com.microsoft":
            atts["rowMajor"] = 0
            atts["computeType"] = "CUBLAS_COMPUTE_32F_FAST_TF32"

        self.added.append(
            make_node(self.main_state.op_gemm, gemm_inputs, gemm_outputs, **atts)
        )
        if do_reshape:
            # One of the inputs had 3 dimensions.
            index = 0 if self.was_reshaped[0] else 1
            fname = f"MatMulReshapeTransposeBack{index}"
            mmshape = node.parent.generate_name(f"{name}_mmshape")

            shape = self.was_reshaped[index]
            if isinstance(shape, tuple):
                self.added.append(
                    make_node(
                        "Constant",
                        [],
                        [mmshape],
                        value=make_tensor(
                            mmshape, TensorProto.INT64, [len(shape)], list(shape)
                        ),
                    )
                )
            elif isinstance(shape, str):
                self.added.append(make_node("Shape", [shape], [mmshape]))
            else:
                raise TypeError(
                    f"Unexpected shape={shape} in was_reshaped={self.was_reshaped}."
                )

            self.added.append(
                make_node(
                    fname,
                    [gemm_outputs[0], mmshape],
                    node.outputs,
                    perm=[1, 0],
                    domain=self.main_state.domain_dq,
                    name=node.parent.generate_node_name("MMRTB"),
                )
            )
            if (
                self.main_state.domain_dq,
                fname,
            ) not in self.main_state.local_functions:
                # use local functions
                self.main_state.local_functions[self.main_state.domain_dq, fname] = (
                    make_matmul_reshape_transpose_back_function_proto(
                        domain=self.main_state.domain_dq,
                        opset=self.main_state.opset,
                        index=index,
                    )
                )

            elif self.was_reshaped[1]:
                raise NotImplementedError(
                    f"Shape is {shape}. This case is not implemented yet."
                )

        self.removed.append(node)
        opsets = {self.main_state.domain_gemm: 1}
        if self.main_state.domain_dq != "":
            opsets.update({self.main_state.domain_dq: 1})
        return opsets


def quantize_float8_matmul(
    node: Node,
    elem_type: int,
    output_type: int,
    version: str,
    opset: int,
    local_functions: Optional[Dict[str, FunctionProto]] = None,
    index_transpose: int = 2,
    domain_ops: Optional[Dict[str, str]] = None,
    quantize_options: QuantizeOptions = QuantizeOptions.NONE,
) -> Optional[TransformResults]:
    """
    Quantizes matrix multiplications.

    :param node: matrix multiplication
    :param elem_type: float 8 type to quantize into
    :param output_type: output type, result of the quantization
    :param version: `'onnxruntime'` to use operators from onnx and onnxruntime,
        `'onnx-extended'` to use experimental operators
    :param opset: main opset (used to specify local functions)
    :param local_functions: None to avoid using local functions,
        otherwise a dictionary with the existing local functions to
        add to the model
    :param quiet: True to silently skip failing nodes
    :param index_transpose: which input to transpose before calling gemm:
        0 (none), 1 (first), 2 (second), 3 for both
    :param domain_ops: domain to use for operators used as keys in the dictionary
    :param quantize_options: see :class:`QuantizeOptions`
    :return: nodes to remove, nodes to add, new opsets

    **About tensors with more than two dimensions.**

    Gemm only supports matrices or 2D tensors.
    When one input does not follow that schema,
    it uses the following tricks. First one when the first input
    has more than one dimension. We could describe that
    with the Einstein summation. We want `abij, jk -> abik`.

    ::

        abij, jk -> abik
        abij ~> Cj
        Cj, jk -> Ck      # gemm
        Ck ~> abik

    In python:

    .. runpython::
        :showcode:

        import numpy as np

        a = np.arange(16).reshape((2, 2, 2, 2))
        b = np.arange(4).reshape((2, 2)) * 10

        expected = a @ b

        a2 = a.reshape((-1, a.shape[-1]))
        b2 = b
        res = a2 @ b2
        final = res.reshape(a.shape[:-2] + (-1, res.shape[-1]))

        print("------")
        print(a)
        print(b)
        print("------")
        print(expected)
        print("------")
        print(final)
        assert expected.tolist() == final.tolist()

    For the other, we use the trick `a @ b = (b' @ a')'`.
    Second one when the second input has more than one dimension.
    We want `ij, abjk -> abik`.

    ::

        ij, abjk -> abik
        abjk ~> Cjk ~> jCk ~> jD
        ij, jD -> iD      # gemm
        iD ~> iCk ~> Cik ~> abik

    .. runpython::
        :showcode:

        import numpy as np

        a = np.arange(4).reshape((2, 2)) * 10
        b = np.arange(16).reshape((2, 2, 2, 2))

        expected = a @ b

        a2 = a
        b2 = (
            b.reshape((-1,) + b.shape[-2:])
            .transpose((1, 0, 2))
            .reshape((b.shape[-2], -1))
        )
        res = a2 @ b2
        final = (
            res.reshape(a.shape[0], -1, b.shape[-1])
            .transpose((1, 0, 2))
            .reshape(b.shape[:-2] + (-1, b.shape[-1]))
        )

        print("------")
        print(a)
        print(b)
        print("------")
        print(expected)
        print("------")
        print(final)
        assert expected.tolist() == final.tolist()

    Both sides cannot have more than two dimensions in the current implementation.
    """
    main_state = _MainQuantizeState(
        version, domain_ops, local_functions, opset, index_transpose
    )

    if node.op_type == "MatMul":
        quantize_state = _QuantizeState(logger, quantize_options, main_state)
        for index, name in enumerate(node.inputs):
            do_transpose = bool((1 << index) & index_transpose)
            if node.parent.is_constant(name):
                # Quantized constant weights
                quantize_state.quantize_connstant_weights(
                    node, name, elem_type, index, do_transpose
                )
            else:
                quantize_state.quantize_dynamic(
                    node, name, elem_type, index, do_transpose
                )

        opsets = quantize_state.finalize(node, name, output_type)

        return TransformResults(
            removed_nodes=quantize_state.removed,
            added_nodes=quantize_state.added,
            new_opsets=opsets,
            local_functions=main_state.local_functions,
        )

    raise NotImplementedError(
        f"Quantization into float 8 not yet implemented for {node.op_type!r}."
    )


def quantize_float8(
    graph: Graph,
    elem_type: int = TensorProto.FLOAT8E4M3FN,
    output_type: int = TensorProto.FLOAT,
    early_stop: int = -1,
    version: str = "onnxruntime",
    quiet: bool = False,
    index_transpose: int = 2,
    domain_ops: Optional[Dict[str, str]] = None,
    exceptions: Optional[List[Dict[str, str]]] = None,
    quantize_options: QuantizeOptions = QuantizeOptions.NONE,
) -> Optional[Graph]:
    """
    Transforms a graph to introduce quantized weights.
    This transformation requires opset 20. The graph is
    upgraded if the main opset is below. It is better to do
    it before calling this function.

    :param graph: Graph
    :param elem_type: quantization type
    :param output_type: output type
    :param early_stop: -1 to go through all nodes or a value `n > 0`
        to stop after n changes
    :param version: `'onnxruntime'` to use operators from onnx and onnxruntime,
        `'onnx-extended'` to use experimental operators
    :param quiet: catch exception and silently skip failing nodes
    :param index_transpose: which input to transpose before calling gemm:
        0 (none), 1 (first), 2 (second), 3 for both
    :param domain_ops: domain to use for operators used as keys in the dictionary
    :param exceptions: exclude nodes from the quantization,
        `[{"name": "node_name1"}, {"name": "node_name2"}]` will exclude
        these two node names from the quantization
    :param quantize_options: see :class:`QuantizeOptions`
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    The graph is modified inplace.
    Enables the logs gives a better idea of the progress.
    """
    main_opset = graph.get_opset("")
    if main_opset < 19:
        logger.info(
            "[quantize_float8] upgrade model from opset %d to %s", main_opset, 19
        )
        graph.upgrade_opsets({"": 19})
        main_opset = 19

    if graph.functions:
        raise NotImplementedError("Quantization of local functions is not implemented.")
    local_functions = graph.functions.copy()
    n_local_functions = len(local_functions)

    new_opsets = {}
    to_add = []
    n_nodes = len(graph)
    n_changes = 0
    for index, node in enumerate(graph):
        if exceptions is not None:
            # checks if a node is not excluded from the list to convert
            has_matched = False
            for exc in exceptions:
                if node.match(exc):
                    has_matched = True
                    break
            if has_matched:
                continue

        if node.op_type in {"MatMul", "Gemm"}:
            logger.info("[quantize_float8] %d/%d quantize %s", index, n_nodes, node)
            try:
                results = quantize_float8_matmul(
                    node,
                    elem_type,
                    output_type,
                    version=version,
                    opset=main_opset,
                    local_functions=local_functions,
                    index_transpose=index_transpose,
                    domain_ops=domain_ops,
                    quantize_options=quantize_options,
                )
            except (QuantizationError, NotImplementedError) as e:
                if quiet:
                    logger.warning(
                        "[quantize_float8] %d/%d failed to quantize due to %s",
                        index,
                        n_nodes,
                        e,
                    )
                    continue
                raise e

            if len(local_functions) != len(results.local_functions) or id(
                local_functions
            ) != id(results.local_functions):
                raise RuntimeError(
                    f"Inconsistency lenghts: "
                    f"{len(local_functions)} != {len(results.local_functions)}, "
                    f"ids: {id(local_functions)} != {id(results.local_functions)}."
                )

            if results is None:
                continue
            rem, add = results.removed_nodes, results.added_nodes
            to_add.append((rem, add))
            if results.new_opsets:
                n_changes += 1
                new_opsets.update(results.new_opsets)
            if early_stop > 0 and n_changes >= early_stop:
                break

    if not to_add:
        return None

    for rem, add in to_add:
        for r in rem:
            logger.debug("[quantize_float8] del %s", r)
        added = graph.replace_nodes([r.index for r in rem], add, new_opsets)
        for a in added:
            logger.debug("[quantize_float8] add %s", a)

    graph.simplify()
    if local_functions is not None and len(local_functions) > n_local_functions:
        graph.add_functions(local_functions.values())
    return graph


def _cast_constant(
    node: Node,
    from_type: int,
    to_type: int,
) -> Optional[TransformResults]:
    """
    Converts a node if it is a tensor of element type *from_type*
    and cast it into *to_type*.

    :param node: node
    :param from_type: element type to replace
    :param to_type: type to cast into
    :return: TransformResults
    """
    op_type = node.op_type

    if op_type in {"Constant", "initializer"}:
        cst = node.get_tensor()
        if cst.data_type != from_type:
            return None

        arr = to_array_extended(cst)
        try:
            cast = Cast.eval(arr, to=to_type)
        except SchemaError:
            # cast does not work
            np_type = tensor_dtype_to_np_dtype(to_type)
            cast = arr.astype(np_type)
        if op_type == "initializer":
            return TransformResults(
                removed_nodes=[node],
                added_nodes=[from_array_extended(cast, name=node.proto.name)],
            )
        if op_type == "Constant":
            return TransformResults(
                removed_nodes=[node],
                added_nodes=[
                    make_node(
                        "Constant",
                        [],
                        node.outputs,
                        value=from_array_extended(cast, name=node.outputs[0]),
                    )
                ],
            )
    if op_type in {"input", "output"}:
        if isinstance(node.proto, str):
            # A FunctionProto input, nothing to do.
            return None
        ttype = node.proto.type
        if not ttype.tensor_type or ttype.tensor_type.elem_type != from_type:
            return None
        return TransformResults(
            removed_nodes=[node],
            added_nodes=[
                make_tensor_value_info(
                    node.proto.name,
                    to_type,
                    shape=_get_shape(ttype),
                    doc_string=node.proto.doc_string,
                    shape_denotation=ttype.denotation,
                )
            ],
        )
    if op_type in {"Cast"}:
        to = node.getattr("to", int)
        if to != from_type:
            return None
        return TransformResults(
            removed_nodes=[node],
            added_nodes=[make_node("Cast", node.inputs, node.outputs, to=to_type)],
        )

    raise RuntimeError(f"Unexpected node type {op_type!r}.")


def cast_constant(
    graph: Graph,
    from_type: int = TensorProto.FLOAT,
    to_type: int = TensorProto.FLOAT16,
    quiet: bool = False,
) -> Optional[Graph]:
    """
    Converts all constants and initializers to the same type.
    It also modifies the input.

    :param graph: Graph
    :param from_type: type of the constants to convert
    :param to_type: new type for the constants
    :param quiet: catch exception and silently skip failing nodes
    :return: Graph or None if not modified

    Transformation are logged with logger `onnx-extended/transformer`.
    The graph is modified inplace.
    Enables the logs gives a better idea of the progress.
    """
    if graph.functions:
        raise NotImplementedError("Conversion of local functions is not implemented.")

    to_add = []
    n_nodes = len(graph)
    for index, node in enumerate(graph):
        if node.op_type in {"Constant", "initializer", "input", "output", "Cast"}:
            logger.info("[cast_constant] %d/%d convert %s", index, n_nodes, node)
            results = _cast_constant(node, from_type, to_type)
            if results is None:
                continue
            rem, add = results.removed_nodes, results.added_nodes
            to_add.append((rem, add))

    if not to_add:
        return None

    for rem, add in to_add:
        for r in rem:
            logger.debug("[cast_constant] del %s", r)
        added = graph.replace_nodes([r.index for r in rem], add)
        for a in added:
            logger.debug("[cast_constant] add %s", a)

    return graph.simplify()
