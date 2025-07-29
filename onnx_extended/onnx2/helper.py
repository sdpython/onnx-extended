# source: https://github.com/onnx/onnx/blob/main/onnx/helper.py
import collections
import functools
import math
import numbers
from typing import Any, NamedTuple, Sequence
import numpy as np
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


class TensorDtypeMap(NamedTuple):
    np_dtype: np.dtype
    storage_dtype: int
    name: str


TENSOR_TYPE_MAP: dict[int, TensorDtypeMap] = {
    int(TensorProto.FLOAT): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
    ),
    int(TensorProto.UINT8): TensorDtypeMap(
        np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
    ),
    int(TensorProto.INT8): TensorDtypeMap(
        np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT8"
    ),
    int(TensorProto.UINT16): TensorDtypeMap(
        np.dtype("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
    ),
    int(TensorProto.INT16): TensorDtypeMap(
        np.dtype("int16"), int(TensorProto.INT32), "TensorProto.INT16"
    ),
    int(TensorProto.INT32): TensorDtypeMap(
        np.dtype("int32"), int(TensorProto.INT32), "TensorProto.INT32"
    ),
    int(TensorProto.INT64): TensorDtypeMap(
        np.dtype("int64"), int(TensorProto.INT64), "TensorProto.INT64"
    ),
    int(TensorProto.BOOL): TensorDtypeMap(
        np.dtype("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
    ),
    int(TensorProto.FLOAT16): TensorDtypeMap(
        np.dtype("float16"), int(TensorProto.INT32), "TensorProto.FLOAT16"
    ),
    int(TensorProto.DOUBLE): TensorDtypeMap(
        np.dtype("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
    ),
    int(TensorProto.COMPLEX64): TensorDtypeMap(
        np.dtype("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
    ),
    int(TensorProto.COMPLEX128): TensorDtypeMap(
        np.dtype("complex128"),
        int(TensorProto.DOUBLE),
        "TensorProto.COMPLEX128",
    ),
    int(TensorProto.UINT32): TensorDtypeMap(
        np.dtype("uint32"), int(TensorProto.UINT64), "TensorProto.UINT32"
    ),
    int(TensorProto.UINT64): TensorDtypeMap(
        np.dtype("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
    ),
    int(TensorProto.STRING): TensorDtypeMap(
        np.dtype("object"), int(TensorProto.STRING), "TensorProto.STRING"
    ),
}


def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Converts a TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        numpy's data_type
    """
    return TENSOR_TYPE_MAP[tensor_dtype].np_dtype


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
    """Creates an empty tensor value info."""
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
    """Returns the attribute value whatever the type is."""
    if attr.ref_attr_name:
        raise ValueError(f"Cannot get value of reference attribute: {attr}")
    if int(attr.type) == AttributeProto.FLOAT:
        return attr.f
    if int(attr.type) == AttributeProto.INT:
        return attr.i
    if int(attr.type) == AttributeProto.STRING:
        return attr.s
    if int(attr.type) == AttributeProto.TENSOR:
        return attr.t
    if int(attr.type) == AttributeProto.SPARSE_TENSOR:
        return attr.sparse_tensor
    if int(attr.type) == AttributeProto.GRAPH:
        return attr.g
    if int(attr.type) == AttributeProto.FLOATS:
        return list(attr.floats)
    if int(attr.type) == AttributeProto.INTS:
        return list(attr.ints)
    if int(attr.type) == AttributeProto.STRINGS:
        return list(attr.strings)
    if int(attr.type) == AttributeProto.TENSORS:
        return list(attr.tensors)
    if int(attr.type) == AttributeProto.SPARSE_TENSORS:
        return list(attr.sparse_tensors)
    if int(attr.type) == AttributeProto.GRAPHS:
        return list(attr.graphs)
    if int(attr.type) == AttributeProto.UNDEFINED:
        return None
    raise ValueError(f"Unsupported ONNX attribute {attr.type} in {attr}")


@functools.lru_cache(None)
def tensor_dtype_to_field(tensor_dtype: int) -> str:
    """
    Converts a TensorProto's data_type to corresponding field name for storage.
    It can be used while making tensors.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        field name
    """
    storage_tensor_type_to_field = {
        int(TensorProto.FLOAT): "float_data",
        int(TensorProto.INT32): "int32_data",
        int(TensorProto.INT64): "int64_data",
        int(TensorProto.DOUBLE): "double_data",
        int(TensorProto.UINT32): "uint64_data",
        int(TensorProto.UINT64): "uint64_data",
        int(TensorProto.STRING): "string_data",
    }
    return storage_tensor_type_to_field[TENSOR_TYPE_MAP[tensor_dtype].storage_dtype]


def make_tensor(
    name: str,
    data_type: int,
    dims: Sequence[int],
    vals: Sequence[int | float] | bytes | np.ndarray,
    raw: bool = False,
) -> TensorProto:
    """
    Makes a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.

    :param name: tensor name
    :param data_type: a value such as onnx.TensorProto.FLOAT
    :param dims: shape
    :param vals: values
    :param raw: if True, vals contains the serialized content of the tensor,
        otherwise, vals should be a list of values
        of the type defined by ``data_type``.
    :return: TensorProto
    """
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    tensor.dims.extend(dims)

    if data_type == TensorProto.STRING and raw:
        raise TypeError("Can not use raw_data to store string type.")

    np_dtype = tensor_dtype_to_np_dtype(data_type)

    if raw:
        # NumPy doesn't have INT4/FP4. It is packed in couples to UINT8 buffers.
        if data_type in {TensorProto.UINT4, TensorProto.INT4, TensorProto.FLOAT4E2M1}:
            expected_size_bytes = 0.5
        else:
            expected_size_bytes = np_dtype.itemsize
        expected_size_bytes *= math.prod(dims)
        expected_size_bytes = math.ceil(expected_size_bytes)
        if isinstance(vals, np.ndarray):
            raw_data = vals.tobytes()
        elif isinstance(vals, bytes):
            raw_data = vals
        else:
            raise TypeError(
                f"Raw data must be bytes or numpy.ndarray, but got {type(vals)}."
            )
        if len(raw_data) != expected_size_bytes:
            raise ValueError(
                f"Raw data size does not match tensor's size. "
                f"Expected {expected_size_bytes} bytes, "
                f"but got {len(raw_data)} bytes."
            )
        tensor.raw_data = raw_data
        return tensor

    assert not raw, "Bug: raw should be False at this point."

    if data_type == TensorProto.STRING:
        vals = np.array(vals).flatten()
        if len(vals) != 0:
            vals = np.vectorize(_to_bytes)(vals)
    else:
        vals = np.asarray(vals, dtype=np_dtype).flatten()

    if data_type == TensorProto.COMPLEX128:
        vals = vals.view(np.float64)
    elif data_type == TensorProto.COMPLEX64:
        vals = vals.view(np.float32)
    elif data_type in {TensorProto.BFLOAT16, TensorProto.FLOAT16}:
        vals = vals.view(np.uint16)
    elif data_type in {
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
        TensorProto.FLOAT8E8M0,
    }:
        vals = vals.view(np.uint8)
    elif data_type == TensorProto.BOOL:
        vals = vals.astype(np.uint8)
    elif data_type >= 16:
        raise AssertionError(f"Unexpected data_type={data_type}.")

    field = tensor_dtype_to_field(data_type)
    getattr(tensor, field).extend(vals)
    return tensor


def make_sparse_tensor(
    values: TensorProto, indices: TensorProto, dims: Sequence[int]
) -> SparseTensorProto:
    """Construct a SparseTensorProto

    Args:
        values (TensorProto): the values
        indices (TensorProto): the indices
        dims: the shape

    Returns:
        SparseTensorProto
    """
    sparse = SparseTensorProto()
    sparse.values.CopyFrom(values)
    sparse.indices.CopyFrom(indices)
    sparse.dims.extend(dims)
    return sparse
