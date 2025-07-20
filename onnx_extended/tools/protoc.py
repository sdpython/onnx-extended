import re
from typing import List, Tuple


TYPE_MAP = {
    "int32": "int32_t",
    "int64": "int64_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "string": "std::string",
    "bool": "bool",
    "float": "float",
    "double": "double",
    "bytes": "std::string",  # simplified
}


class Enum:
    def __init__(self, name: str):
        self.name = name
        self.values: List[Tuple[str, str]] = []


class Message:
    def __init__(self, name: str):
        self.name = name
        self.fields: List[Tuple[str, str, str]] = []  # (type, name, field_number)
        self.enums: List[Enum] = []
        self.nested_messages: List["Message"] = []  # noqa: UP037
        self.oneofs: List[Tuple[str, List[Tuple[str, str, str]]]] = (
            []
        )  # (oneof_name, [(type, name, field_number)])


def remove_comments(proto: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", proto, flags=re.DOTALL | re.MULTILINE)


def parse_enums(block: str) -> List[Enum]:
    enums = []
    enum_pattern = re.compile(r"enum\s+(\w+)\s*{(.*?)}", re.DOTALL)
    for match in enum_pattern.finditer(block):
        enum = Enum(match.group(1))
        body = match.group(2)
        value_pattern = re.compile(r"(\w+)\s*=\s*(\d+);")
        enum.values.extend(value_pattern.findall(body))
        enums.append(enum)
    return enums


def parse_messages(proto: str) -> List[Message]:
    def parse_block(block: str) -> List[Message]:
        messages = []
        message_pattern = re.compile(r"message\s+(\w+)\s*{")
        pos = 0
        while True:
            match = message_pattern.search(block, pos)
            if not match:
                break
            name = match.group(1)
            start = match.end()
            brace_count = 1
            i = start
            while i < len(block) and brace_count > 0:
                if block[i] == "{":
                    brace_count += 1
                elif block[i] == "}":
                    brace_count -= 1
                i += 1
            body = block[start : i - 1].strip()
            message = Message(name)

            # Parse normal fields
            field_pattern = re.compile(
                r"(optional|repeated)?\s*(\w+)\s+(\w+)\s*=\s*(\d+);"
            )
            for qualifier, type_, name_, field_number in field_pattern.findall(body):
                cpp_type = TYPE_MAP.get(type_, type_)
                if qualifier == "repeated":
                    cpp_type = f"std::vector<{cpp_type}>"
                elif qualifier == "optional":
                    cpp_type = f"std::optional<{cpp_type}>"
                message.fields.append((cpp_type, name_, field_number))

            # Parse oneof
            oneof_pattern = re.compile(r"oneof\s+(\w+)\s*{(.*?)}", re.DOTALL)
            for oneof_match in oneof_pattern.findall(body):
                oneof_name = oneof_match[0]
                oneof_body = oneof_match[1]
                oneof_fields = []
                ofield_pattern = re.compile(r"(\w+)\s+(\w+)\s*=\s*(\d+);")
                for type_, name_, number in ofield_pattern.findall(oneof_body):
                    cpp_type = TYPE_MAP.get(type_, type_)
                    oneof_fields.append((cpp_type, name_, number))
                message.oneofs.append((oneof_name, oneof_fields))

            # Enums and nested messages
            message.enums = parse_enums(body)
            message.nested_messages = parse_block(body)
            messages.append(message)
            pos = i
        return messages

    return parse_block(proto)


def parse_top_level_enums(proto: str) -> List[Enum]:
    return parse_enums(proto)


def generate_cpp_enum(enum: Enum, indent=0) -> str:
    prefix = " " * indent
    lines = [f"{prefix}enum class {enum.name} {{"]
    for name, number in enum.values:
        lines.append(f"{prefix}    {name} = {number},")
    lines.append(f"{prefix}}};")
    return "\n".join(lines)


def generate_cpp_struct(message: Message, indent=0) -> str:
    prefix = " " * indent
    lines = [f"{prefix}struct {message.name} {{"]

    # Normal fields
    for type_, name, number in message.fields:
        lines.append(f"{prefix}    {type_} {name};  // field {number}")

    # Oneof fields (as std::variant)
    for oneof_name, variants in message.oneofs:
        types = ", ".join([t for t, _, _ in variants])
        lines.append(f"{prefix}    std::variant<{types}> {oneof_name};  // oneof")
        for t, name, number in variants:
            lines.append(f"{prefix}    // oneof field: {t} {name} = {number}")

    # Nested enums
    for enum in message.enums:
        lines.append("")
        lines.append(generate_cpp_enum(enum, indent + 4))

    # Nested messages
    for nested in message.nested_messages:
        lines.append("")
        lines.append(generate_cpp_struct(nested, indent + 4))

    lines.append(f"{prefix}}};")
    return "\n".join(lines)


def generate_cpp_header(
    messages: List[Message], enums: List[Enum], header_name: str
) -> str:
    guard = f"__{header_name.upper()}_H__"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <string>",
        "#include <vector>",
        "#include <cstdint>",
        "#include <optional>",
        "#include <variant>",
        "",
    ]

    for enum in enums:
        lines.append(generate_cpp_enum(enum))
        lines.append("")

    for msg in messages:
        lines.append(generate_cpp_struct(msg))
        lines.append("")

    lines.append(f"#endif // {guard}")
    return "\n".join(lines)


def parse_proto(content: str) -> str:
    """
    Parses a string coming from a proto file defining structures for protobuf.

    :param content: content
    :return: C++ header
    """
    content = remove_comments(content)
    messages = parse_messages(content)
    enums = parse_top_level_enums(content)
    header_code = generate_cpp_header(messages, enums, "parsed_proto")
    return header_code
