from . import AttributeProto, SparseTensorProto


class ValidationError(ValueError):
    "Invalid proto"


def check_attribute(att: AttributeProto):
    """Checks an AttributeProto is valid."""
    oneof = [
        att.has_f(),
        att.has_i(),
        att.has_s(),
        att.has_t(),
        att.has_sparse_tensor(),
        att.has_floats(),
        att.has_ints(),
        att.has_strings(),
        att.has_tensors(),
        att.has_sparse_tensors(),
    ]
    if not any(oneof):
        raise ValidationError(f"The attribute has no value: {att}")
    total = sum(int(i) for i in oneof)
    if total != 1:
        raise ValidationError(f"The attribute has more than one value: {att}")


def check_sparse_tensor(sp: SparseTensorProto):
    """Checks a SparseTensorProto is valid."""
    shape = tuple(sp.dims)
    if len(shape) != 2:
        raise ValidationError(f"Only 2D sparse tensors are allowed: {shape}")
