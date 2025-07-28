from . import AttributeProto


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
    assert any(oneof), f"The attribute has no value: {att}"
    total = sum(int(i) for i in oneof)
    assert total == 1, f"The attribute has more than one value: {att}"
