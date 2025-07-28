# source: https://github.com/onnx/onnx/blob/main/onnx/helper.py
import collections
import numbers
from typing import Any, Sequence
from .cpu._onnx2py import (
    AttributeProto,
    OperatorSetIdProto,
    SparseTensorProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
)

_ATTRIBUTE_TYPE_TO_STR: dict[int, str] = {
    k: v for v, k in AttributeProto.AttributeType.items()
}
_ATTRIBUTE_TYPE_INT_TO_STR: dict[int, str] = {
    int(k): v for v, k in AttributeProto.AttributeType.items()
}


def make_operatorsetid(
    domain: str,
    version: int,
) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Args:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid


def make_tensor_type_proto(
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    shape_denotation: list[str] | None = None,
) -> TypeProto:
    """Makes a Tensor TypeProto based on the data type and shape."""
    type_proto = TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = elem_type
    tensor_shape_proto = tensor_type_proto.shape

    if shape is not None:
        tensor_shape_proto.dim.extend([])

        if shape_denotation and len(shape_denotation) != len(shape):
            raise ValueError(
                "Invalid shape_denotation. Must be of the same length as shape."
            )

        for i, d in enumerate(shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    f"Invalid item in shape: {d}. Needs to be of int or str."
                )

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_empty_tensor_value_info(name: str) -> ValueInfoProto:
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    return value_info_proto


def make_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    doc_string: str = "",
    shape_denotation: list[str] | None = None,
) -> ValueInfoProto:
    """Makes a ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.CopyFrom(tensor_type_proto)
    return value_info_proto


def make_sparse_tensor_type_proto(
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    shape_denotation: list[str] | None = None,
) -> TypeProto:
    """Makes a SparseTensor TypeProto based on the data type and shape."""
    type_proto = TypeProto()
    sparse_tensor_type_proto = type_proto.sparse_tensor_type
    sparse_tensor_type_proto.elem_type = elem_type
    sparse_tensor_shape_proto = sparse_tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        sparse_tensor_shape_proto.dim.extend([])

        if shape_denotation and len(shape_denotation) != len(shape):
            raise ValueError(
                "Invalid shape_denotation. Must be of the same length as shape."
            )

        for i, d in enumerate(shape):
            dim = sparse_tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    f"Invalid item in shape: {d}. Needs to be of int or text."
                )

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_sparse_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    doc_string: str = "",
    shape_denotation: list[str] | None = None,
) -> ValueInfoProto:
    """Makes a SparseTensor ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    sparse_tensor_type_proto = make_sparse_tensor_type_proto(
        elem_type, shape, shape_denotation
    )
    value_info_proto.type.sparse_tensor_type.CopyFrom(
        sparse_tensor_type_proto.sparse_tensor_type
    )
    return value_info_proto


def make_sequence_type_proto(
    inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes a sequence TypeProto."""
    type_proto = TypeProto()
    type_proto.sequence_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_optional_type_proto(
    inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes an optional TypeProto."""
    type_proto = TypeProto()
    type_proto.optional_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_map_type_proto(
    key_type: int,
    value_type: TypeProto,
) -> TypeProto:
    """Makes a map TypeProto."""
    type_proto = TypeProto()
    type_proto.map_type.key_type = key_type
    type_proto.map_type.value_type.CopyFrom(value_type)
    return type_proto


def make_value_info(
    name: str,
    type_proto: TypeProto,
    doc_string: str = "",
) -> ValueInfoProto:
    """Makes a ValueInfoProto with the given type_proto."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    value_info_proto.type.CopyFrom(type_proto)
    return value_info_proto


def make_tensor_sequence_value_info(
    name: str,
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    doc_string: str = "",
    elem_shape_denotation: list[str] | None = None,
) -> ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, elem_shape_denotation)
    sequence_type_proto = make_sequence_type_proto(tensor_type_proto)
    value_info_proto.type.sequence_type.CopyFrom(sequence_type_proto.sequence_type)

    return value_info_proto


def _attr_type_to_str(attr_type: int) -> str:
    """Convert AttributeProto type to string.

    Args:
        attr_type: AttributeProto type.

    Returns:
        String representing the supplied attr_type.
    """
    if attr_type in _ATTRIBUTE_TYPE_TO_STR:
        return _ATTRIBUTE_TYPE_TO_STR[attr_type]
    if isinstance(attr_type, int) and attr_type in _ATTRIBUTE_TYPE_INT_TO_STR:
        return _ATTRIBUTE_TYPE_INT_TO_STR[attr_type]
    return AttributeProto.AttributeType.keys()[0]


def _to_bytes(value: str | bytes) -> bytes:
    """Coerce a string (or bytes) value into UTF-8 bytes."""
    if isinstance(value, str):
        return value.encode("utf-8")
    return value


def make_attribute(
    key: str,
    value: Any,
    doc_string: str | None = None,
    attr_type: int | None = None,
) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    if isinstance(attr_type, int):
        attr_type = AttributeProto.AttributeType(attr_type)
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    # Singular cases
    if isinstance(value, numbers.Integral):
        attr.i = int(value)
        attr.type = AttributeProto.INT
    elif isinstance(value, numbers.Real):
        attr.f = float(value)
        attr.type = AttributeProto.FLOAT
    elif isinstance(value, (str, bytes)):
        # Encode strings into utf-8
        attr.s = _to_bytes(value)
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, SparseTensorProto):
        attr.sparse_tensor.CopyFrom(value)
        attr.type = AttributeProto.SPARSE_TENSOR
    # elif isinstance(value, GraphProto):
    #    attr.g.CopyFrom(value)
    #    attr.type = AttributeProto.GRAPH
    # elif isinstance(value, TypeProto):
    #    attr.tp.CopyFrom(value)
    #    attr.type = AttributeProto.TYPE_PROTO
    # Iterable cases
    elif isinstance(value, collections.abc.Iterable):
        value = list(value)
        if len(value) == 0 and attr_type is None:
            raise ValueError(
                f"Could not infer attribute {key!r} type from empty iterator"
            )
        if attr_type is None:
            types = {type(v) for v in value}
            for exp_t, exp_enum in (
                (numbers.Integral, AttributeProto.INTS),
                (numbers.Real, AttributeProto.FLOATS),
                ((str, bytes), AttributeProto.STRINGS),
                (TensorProto, AttributeProto.TENSORS),
                (SparseTensorProto, AttributeProto.SPARSE_TENSORS),
                # (GraphProto, AttributeProto.GRAPHS),
            ):
                if all(issubclass(t, exp_t) for t in types):
                    attr_type = exp_enum
                    break
            if attr_type is None:
                raise ValueError(
                    "Could not infer the attribute type from the "
                    "elements of the passed Iterable value."
                )

        if int(attr_type) == AttributeProto.INTS:
            attr.ints.extend(value)
            attr.type = AttributeProto.INTS
        elif int(attr_type) == AttributeProto.FLOATS:
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif int(attr_type) == AttributeProto.STRINGS:
            attr.strings.extend(_to_bytes(v) for v in value)
            attr.type = AttributeProto.STRINGS
        elif int(attr_type) == AttributeProto.TENSORS:
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif int(attr_type) == AttributeProto.SPARSE_TENSORS:
            attr.sparse_tensors.extend(value)
            attr.type = AttributeProto.SPARSE_TENSORS
        elif int(attr_type) == AttributeProto.GRAPHS:
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        else:
            raise AssertionError(f"Unexpected type={attr_type} for an attribute.")
    else:
        raise TypeError(f"{value!r} is not an accepted attribute value.")

    if attr.type == AttributeProto.AttributeType.UNDEFINED and attr_type != attr.type:
        attr.type = attr_type
    if attr_type is not None and int(attr.type) != int(attr_type):
        raise TypeError(
            f"Inferred attribute type {_attr_type_to_str(attr.type)!r}({attr.type}) "
            f"mismatched with specified type {_attr_type_to_str(attr_type)!r}"
            f"({attr_type})"
        )
    return attr


def make_attribute_ref(
    name: str, attr_type: AttributeProto.AttributeType, doc_string: str | None = None
) -> AttributeProto:
    """
    Makes an AttributeProto holding a reference to the parent "
    function's attribute of given name and type.
    """
    attr = AttributeProto()
    attr.name = name
    attr.type = attr_type
    if doc_string:
        attr.doc_string = doc_string
    return attr


def get_attribute_value(attr: AttributeProto) -> Any:
    if attr.ref_attr_name:
        raise ValueError(f"Cannot get value of reference attribute: {attr}")
    if attr.type == AttributeProto.FLOAT:
        return attr.f
    if attr.type == AttributeProto.INT:
        return attr.i
    if attr.type == AttributeProto.STRING:
        return attr.s
    if attr.type == AttributeProto.TENSOR:
        return attr.t
    if attr.type == AttributeProto.SPARSE_TENSOR:
        return attr.sparse_tensor
    if attr.type == AttributeProto.GRAPH:
        return attr.g
    if attr.type == AttributeProto.TYPE_PROTO:
        return attr.tp
    if attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == AttributeProto.STRINGS:
        return list(attr.strings)
    if attr.type == AttributeProto.TENSORS:
        return list(attr.tensors)
    if attr.type == AttributeProto.SPARSE_TENSORS:
        return list(attr.sparse_tensors)
    if attr.type == AttributeProto.GRAPHS:
        return list(attr.graphs)
    if attr.type == AttributeProto.TYPE_PROTOS:
        return list(attr.type_protos)
    if attr.type == AttributeProto.UNDEFINED:
        return None
    raise ValueError(f"Unsupported ONNX attribute: {attr}")
