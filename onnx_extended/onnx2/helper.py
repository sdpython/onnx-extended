from .cpu._onnx2py import OperatorSetIdProto
from typing import Any, Sequence, Union

VersionRowType = Union[tuple[str, int, int, int], tuple[str, int, int, int, int]]
VersionTableType = list[VersionRowType]
AssignmentBindingType = list[tuple[str, str]]

VERSION_TABLE: VersionTableType = [
    ("1.0", 3, 1, 1),
    ("1.1", 3, 5, 1),
    ("1.1.2", 3, 6, 1),
    ("1.2", 3, 7, 1),
    ("1.3", 3, 8, 1),
    ("1.4.1", 4, 9, 1),
    ("1.5.0", 5, 10, 1),
    ("1.6.0", 6, 11, 2),
    ("1.7.0", 7, 12, 2, 1),
    ("1.8.0", 7, 13, 2, 1),
    ("1.8.1", 7, 13, 2, 1),
    ("1.9.0", 7, 14, 2, 1),
    ("1.10.0", 8, 15, 2, 1),
    ("1.10.1", 8, 15, 2, 1),
    ("1.10.2", 8, 15, 2, 1),
    ("1.11.0", 8, 16, 3, 1),
    ("1.12.0", 8, 17, 3, 1),
    ("1.13.0", 8, 18, 3, 1),
    ("1.13.1", 8, 18, 3, 1),
    ("1.14.0", 9, 19, 3, 1),
    ("1.14.1", 9, 19, 3, 1),
    ("1.15.0", 9, 20, 4, 1),
    ("1.16.0", 10, 21, 5, 1),
    ("1.17.0", 10, 22, 5, 1),
    ("1.18.0", 11, 23, 5, 1),
]

VersionMapType = dict[tuple[str, int], int]


def _create_op_set_id_version_map(table: VersionTableType) -> VersionMapType:
    """Create a map from (opset-domain, opset-version)
    to ir-version from above table."""
    result: VersionMapType = {}

    def process(release_version: str, ir_version: int, *args: Any) -> None:
        del release_version  # Unused
        for pair in zip(["ai.onnx", "ai.onnx.ml", "ai.onnx.training"], args):
            if pair not in result:
                result[pair] = ir_version
                if pair[0] == "ai.onnx.training":
                    result["ai.onnx.preview.training", pair[1]] = ir_version

    for row in table:
        process(*row)
    return result


OP_SET_ID_VERSION_MAP = _create_op_set_id_version_map(VERSION_TABLE)


def find_min_ir_version_for(
    opsetidlist: Sequence[OperatorSetIdProto], ignore_unknown: bool = False
) -> int:
    """Given list of opset ids, determine minimum IR version required.

    Args:
        opsetidlist: A sequence of OperatorSetIdProto.
        ignore_unknown: If True, ignore unknown domain and return default minimum
            version for that domain.

    Returns:
        The minimum IR version required (integer)
    """
    default_min_version = 3

    def find_min(domain: str | None, version: int) -> int:
        key = (domain or "ai.onnx", version)
        if key in OP_SET_ID_VERSION_MAP:
            return OP_SET_ID_VERSION_MAP[key]
        if ignore_unknown:
            return default_min_version
        raise ValueError("Unsupported opset-version.")

    if opsetidlist:
        return max(find_min(x.domain, x.version) for x in opsetidlist)
    return default_min_version  # if no opsets specified


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
