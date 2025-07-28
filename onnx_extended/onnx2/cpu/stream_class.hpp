#pragma once

#include "stream_class.h"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <vector>

// #define DEBUG_READ

#if defined(DEBUG_READ)
#define DEBUG_PRINT(s) printf("%s\n", s);
#define DEBUG_PRINT2(s1, s2) printf("%s%s\n", s1, s2);
#else
#define DEBUG_PRINT(s)
#define DEBUG_PRINT2(s1, s2)
#endif

//////////////////
// macro helpers
////////////////

#define NAME_EXIST_VALUE(name) name_exist_value(_name_##name, has_##name(), ptr_##name())

#define IMPLEMENT_PROTO(cls)                                                                           \
  void cls::CopyFrom(const cls &proto) {                                                               \
    utils::StringWriteStream stream;                                                                   \
    proto.SerializeToStream(stream);                                                                   \
    utils::StringStream read_stream(stream.data(), stream.size());                                     \
    ParseFromStream(read_stream);                                                                      \
  }

//////////////
// macro write
//////////////

#define WRITE_FIELD(stream, name)                                                                      \
  if (has_##name()) {                                                                                  \
    write_field(stream, order_##name(), ref_##name());                                                 \
  }

#define WRITE_ENUM_FIELD(stream, name)                                                                 \
  if (has_##name()) {                                                                                  \
    write_enum_field(stream, order_##name(), ref_##name());                                            \
  }

#define WRITE_REPEATED_FIELD(stream, name)                                                             \
  if (has_##name()) {                                                                                  \
    write_repeated_field(stream, order_##name(), name##_, packed_##name());                            \
  }

#define WRITE_OPTIONAL_PROTO_FIELD(stream, name)                                                       \
  if (has_##name()) {                                                                                  \
    write_optional_proto_field(stream, order_##name(), name##_optional());                             \
  }

/////////////
// macro read
/////////////

#define READ_BEGIN(stream, cls)                                                                        \
  DEBUG_PRINT("+ read begin " #cls)                                                                    \
  while (stream.not_end()) {                                                                           \
    utils::FieldNumber field_number = stream.next_field();                                             \
    DEBUG_PRINT2("  = field number ", field_number.string().c_str())                                   \
    if (field_number.field_number == 0) {                                                              \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);                \
    }

#define READ_END(stream, cls)                                                                          \
  else {                                                                                               \
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);             \
  }                                                                                                    \
  }                                                                                                    \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(stream, name)                                                                       \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + field " #name)                                                                    \
    read_field(stream, field_number.wire_type, name##_, #name);                                        \
    DEBUG_PRINT("  - field " #name)                                                                    \
  }

#define READ_OPTIONAL_PROTO_FIELD(stream, name)                                                        \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + optional field " #name)                                                           \
    read_optional_proto_field(stream, field_number.wire_type, name##_, #name);                         \
    DEBUG_PRINT("  - optional field " #name)                                                           \
  }

#define READ_ENUM_FIELD(stream, name)                                                                  \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + enum " #name)                                                                     \
    read_enum_field(stream, field_number.wire_type, name##_, #name);                                   \
    DEBUG_PRINT("  - enum " #name)                                                                     \
  }

#define READ_REPEATED_FIELD(stream, name)                                                              \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + repeat " #name)                                                                   \
    read_repeated_field(stream, field_number.wire_type, name##_, #name, packed_##name());              \
    DEBUG_PRINT("  - repeat " #name)                                                                   \
  }

using namespace onnx_extended_helpers;

namespace onnx2 {

////////
// write
////////

template <typename T> void write_field(utils::BinaryWriteStream &stream, int order, const T &field) {
  utils::StringWriteStream local;
  field.SerializeToStream(local);
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string_stream(local);
}

template <typename T>
void write_optional_proto_field(utils::BinaryWriteStream &stream, int order,
                                const utils::OptionalField<T> &field) {
  if (field.has_value()) {
    utils::StringWriteStream local;
    (*field).SerializeToStream(local);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const utils::String &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<uint64_t> &field) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(*field);
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<int64_t> &field) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(static_cast<uint64_t>(*field));
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<int32_t> &field) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(static_cast<uint64_t>(*field));
  }
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const int64_t &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_int64(field);
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const uint64_t &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(field);
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const int32_t &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_int32(field);
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const double &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE); // FIELD_FIXED64);
  stream.write_double(field);
}

template <> void write_field(utils::BinaryWriteStream &stream, int order, const float &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE); // FIELD_FIXED32);
  stream.write_float(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const std::vector<uint8_t> &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  utils::BorrowedWriteStream local(field.data(), field.size());
  stream.write_string_stream(local);
}

template <typename T>
void write_enum_field(utils::BinaryWriteStream &stream, int order, const T &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(static_cast<uint64_t>(field));
}

template <typename T>
void write_repeated_field(utils::BinaryWriteStream &stream, int order, const std::vector<T> &field,
                          bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (const auto &d : field) {
    utils::StringWriteStream local;
    d.SerializeToStream(local);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const std::vector<utils::String> &field, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (const auto &d : field) {
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string(d);
  }
}

#define WRITE_REPEATED_FIELD_IMPL(type, unpack_method)                                                 \
  template <>                                                                                          \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                               \
                            const std::vector<type> &field, bool is_packed) {                          \
    if (is_packed) {                                                                                   \
      stream.write_field_header(order, FIELD_FIXED_SIZE);                                              \
      stream.write_variant_uint64(field.size() * sizeof(type));                                        \
      for (const auto &d : field) {                                                                    \
        stream.write_packed_element(d);                                                                \
      }                                                                                                \
    } else {                                                                                           \
      for (const auto &d : field) {                                                                    \
        stream.write_field_header(order, FIELD_FIXED_SIZE);                                            \
        stream.unpack_method(d);                                                                       \
      }                                                                                                \
    }                                                                                                  \
  }

WRITE_REPEATED_FIELD_IMPL(double, write_double)
WRITE_REPEATED_FIELD_IMPL(float, write_float)

#define WRITE_REPEATED_FIELD_IMPL_INT(type)                                                            \
  template <>                                                                                          \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                               \
                            const std::vector<type> &field, bool is_packed) {                          \
    if (is_packed) {                                                                                   \
      stream.write_field_header(order, FIELD_FIXED_SIZE);                                              \
      stream.write_variant_uint64(field.size());                                                       \
      for (const auto &d : field) {                                                                    \
        stream.write_variant_uint64(static_cast<uint64_t>(d));                                         \
      }                                                                                                \
    } else {                                                                                           \
      for (const auto &d : field) {                                                                    \
        stream.write_field_header(order, FIELD_VARINT);                                                \
        stream.write_variant_uint64(static_cast<uint64_t>(d));                                         \
      }                                                                                                \
    }                                                                                                  \
  }

WRITE_REPEATED_FIELD_IMPL_INT(uint64_t)
WRITE_REPEATED_FIELD_IMPL_INT(int64_t)
WRITE_REPEATED_FIELD_IMPL_INT(int32_t)

template <typename T>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const utils::RepeatedField<T> &field, bool is_packed) {
  write_repeated_field(stream, order, field.values, is_packed);
}

///////
// read
///////

template <typename T>
void read_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  field.ParseFromStream(dim_buf);
}

template <typename T>
void read_optional_proto_field(utils::BinaryStream &stream, int wire_type,
                               utils::OptionalField<T> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  field.set_empty_value();
  (*field).ParseFromStream(dim_buf);
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::RefString &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::String &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int64_t> &field,
                const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int32_t> &field,
                const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, uint64_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int64_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int32_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, float &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, double &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_double();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  uint64_t len = stream.next_uint64();
  field.resize(len);
  memcpy(field.data(), stream.read_bytes(len), len);
}

template <typename T>
void read_enum_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = static_cast<TensorProto::DataType>(stream.next_uint64());
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  T elem;
  elem.ParseFromStream(dim_buf);
  field.emplace_back(elem);
}

template <>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<utils::String> &field,
                         const char *name, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field.emplace_back(utils::String(stream.next_string()));
}

#define READ_REPEATED_FIELD_IMPL(type, unpack_method, unpacked_wire_type)                              \
  template <>                                                                                          \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<type> &field,       \
                           const char *name, bool is_packed) {                                         \
    if (is_packed) {                                                                                   \
      DEBUG_PRINT2("    read packed", name);                                                           \
      EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",   \
                  name, "'");                                                                          \
      uint64_t size = stream.next_uint64();                                                            \
      EXT_ENFORCE(size % sizeof(type) == 0, "unexpected size ", size,                                  \
                  ", it is not a multiple of sizeof(" #type ")");                                      \
      size /= sizeof(type);                                                                            \
      field.resize(size);                                                                              \
      for (size_t i = 0; i < static_cast<size_t>(size); ++i) {                                         \
        stream.next_packed_element(field[i]);                                                          \
      }                                                                                                \
    } else {                                                                                           \
      DEBUG_PRINT2("    read unpacked", name);                                                         \
      EXT_ENFORCE(wire_type == unpacked_wire_type, "unexpected wire_type=", wire_type, " for field '", \
                  name, "'");                                                                          \
      field.push_back(stream.unpack_method());                                                         \
    }                                                                                                  \
  }

READ_REPEATED_FIELD_IMPL(double, next_double, FIELD_FIXED_SIZE)
READ_REPEATED_FIELD_IMPL(float, next_float, FIELD_FIXED_SIZE)

#define READ_REPEATED_FIELD_IMPL_INT(type, unpack_method)                                              \
  template <>                                                                                          \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<type> &field,       \
                           const char *name, bool is_packed) {                                         \
    if (is_packed) {                                                                                   \
      DEBUG_PRINT2("    read packed", name);                                                           \
      EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",   \
                  name, "'");                                                                          \
      uint64_t size = stream.next_uint64();                                                            \
      field.resize(size);                                                                              \
      for (size_t i = 0; i < static_cast<size_t>(size); ++i) {                                         \
        field[i] = stream.unpack_method();                                                             \
      }                                                                                                \
    } else {                                                                                           \
      DEBUG_PRINT2("    read unpacked", name);                                                         \
      EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, \
                  "'");                                                                                \
      field.push_back(stream.unpack_method());                                                         \
    }                                                                                                  \
  }

READ_REPEATED_FIELD_IMPL_INT(int64_t, next_int64)
READ_REPEATED_FIELD_IMPL_INT(int32_t, next_int32)
READ_REPEATED_FIELD_IMPL_INT(uint64_t, next_uint64)

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, utils::RepeatedField<T> &field,
                         const char *name, bool is_packed) {
  read_repeated_field(stream, wire_type, field.values, name, is_packed);
}

////////////////////////////
// serialization into string
////////////////////////////

template <typename T> struct name_exist_value {
  const char *name;
  bool exist;
  const T *value;
  inline name_exist_value(const char *n, bool e, const T *v) : name(n), exist(e), value(v) {}
};

template <typename T> std::string write_as_string(const T &field) { return MakeString(field); }

template <> std::string write_as_string(const utils::String &field) {
  return MakeString("\"", field.as_string(), "\"");
}

template <> std::string write_as_string(const std::vector<uint8_t> &field) {
  const char *hex_chars = "0123456789ABCDEF";
  std::stringstream result;
  for (const auto &b : field) {
    result << hex_chars[b / 16] << hex_chars[b % 16];
  }
  return result.str();
}

template <typename T> std::string write_as_string_vector(const std::vector<T> &field) {
  std::stringstream result;
  result << "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result << field[i];
    if (i + 1 != field.size())
      result << ", ";
  }
  result << "]";
  return result.str();
}

template <typename T> std::string write_as_string_optional(const std::optional<T> &field) {
  if (!field)
    return "null";
  return write_as_string(*field);
}

template <> std::string write_as_string(const std::vector<float> &field) {
  return write_as_string_vector(field);
}

template <> std::string write_as_string(const std::vector<int64_t> &field) {
  return write_as_string_vector(field);
}

template <> std::string write_as_string(const std::vector<uint64_t> &field) {
  return write_as_string_vector(field);
}

template <> std::string write_as_string(const std::vector<double> &field) {
  return write_as_string_vector(field);
}

template <> std::string write_as_string(const std::vector<int32_t> &field) {
  return write_as_string_vector(field);
}

template <> std::string write_as_string(const std::optional<float> &field) {
  return write_as_string_optional(field);
}

template <> std::string write_as_string(const std::optional<int64_t> &field) {
  return write_as_string_optional(field);
}

template <> std::string write_as_string(const std::optional<uint64_t> &field) {
  return write_as_string_optional(field);
}

template <> std::string write_as_string(const std::optional<double> &field) {
  return write_as_string_optional(field);
}

template <> std::string write_as_string(const std::optional<int32_t> &field) {
  return write_as_string_optional(field);
}

template <typename... Args> std::string write_as_string(const Args &...args) {
  std::stringstream result;
  result << "{";

  auto append_arg = [&result, first = true](const auto &arg) mutable {
    if (arg.exist) {
      if (!first) {
        result << ", ";
      }
      first = false;
      result << arg.name;
      result << ": ";
      result << write_as_string(*arg.value);
    }
  };

  (append_arg(args), ...);
  result << "}";
  return result.str();
}

template <typename T>
std::vector<std::string> write_into_vector_string(const char *field_name, const T &field) {
  std::vector<std::string> r = field.SerializeToVectorString();
  if (r.size() <= 1) {
    return {MakeString(field_name, ": ", r.back(), ",")};
  } else {
    std::vector<std::string> rows{MakeString(field_name, ": ")};
    for (size_t i = 0; i < r.size(); ++i) {
      if (i == 0) {
        rows[0] += r[0];
      } else if (i + 1 == r.size()) {
        rows.push_back(MakeString("  ", r[i], ","));
      } else {
        rows.push_back(MakeString("  ", r[i]));
      }
    }
    return rows;
  }
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name, const utils::String &field) {
  return {MakeString(field_name, ": ", write_as_string(field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name, const int64_t &field) {
  return {MakeString(field_name, ": ", write_as_string(field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name, const uint64_t &field) {
  return {MakeString(field_name, ": ", write_as_string(field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name, const int32_t &field) {
  return {MakeString(field_name, ": ", write_as_string(field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const TensorProto::DataType &field) {
  return {MakeString(field_name, ": ", write_as_string(static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const std::vector<uint8_t> &field) {
  return {MakeString(field_name, ": ", write_as_string(field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<utils::String> &field) {
  std::vector<std::string> rows{MakeString(field_name, ": [")};
  for (const auto &p : field) {
    auto r = p.as_string();
    rows.push_back(MakeString("  ", r, ","));
  }
  rows.push_back("],");
  return rows;
}

template <typename T>
std::vector<std::string> write_into_vector_string_repeated(const char *field_name,
                                                           const utils::RepeatedField<T> &field) {
  std::vector<std::string> rows;
  if (field.size() >= 10) {
    rows.push_back(MakeString(field_name, ": ["));
    for (const auto &p : field) {
      rows.push_back(MakeString("  ", p, ","));
    }
    rows.push_back("],");
  } else {
    std::vector<std::string> r;
    for (const auto &p : field) {
      r.push_back(MakeString(p));
    }
    rows.push_back(MakeString(field_name, ": [", utils::join_string(r, ", "), "],"));
  }
  rows.push_back("],");
  return rows;
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<uint64_t> &field) {
  return write_into_vector_string_repeated(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<int64_t> &field) {
  return write_into_vector_string_repeated(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<int32_t> &field) {
  return write_into_vector_string_repeated(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<float> &field) {
  return write_into_vector_string_repeated(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::RepeatedField<double> &field) {
  return write_into_vector_string_repeated(field_name, field);
}

template <typename T>
std::vector<std::string> write_into_vector_string_optional(const char *field_name,
                                                           const utils::OptionalField<T> &field) {
  if (field.has_value()) {
    return {MakeString(field_name, ": ", write_as_string(field.value), ",")};
  } else {
    return {MakeString(field_name, ": null,")};
  }
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::OptionalField<int64_t> &field) {
  return write_into_vector_string_optional(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::OptionalField<uint64_t> &field) {
  return write_into_vector_string_optional(field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(const char *field_name,
                                                  const utils::OptionalField<int32_t> &field) {
  return write_into_vector_string_optional(field_name, field);
}

template <typename... Args>
std::vector<std::string> write_proto_into_vector_string(const Args &...args) {
  std::vector<std::string> rows{"{"};
  auto append_arg = [&rows, first = true](const auto &arg) mutable {
    if (arg.exist) {
      std::vector<std::string> r = write_into_vector_string(arg.name, *arg.value);
      for (const auto &s : r) {
        rows.push_back("  " + s);
      }
    }
  };
  (append_arg(args), ...);
  rows.push_back("},");
  return rows;
}

} // namespace onnx2
