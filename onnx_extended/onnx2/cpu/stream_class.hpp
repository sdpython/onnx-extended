#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

// #define DEBUG_READ

#if defined(DEBUG_READ)
#define DEBUG_PRINT(s) printf("%s\n", s);
#define DEBUG_PRINT2(s1, s2) printf("%s%s\n", s1, s2);
#else
#define DEBUG_PRINT(s)
#define DEBUG_PRINT2(s1, s2)
#endif

#define WRITE_FIELD(stream, name)                                                              \
  if (has_##name()) {                                                                          \
    write_field(stream, order_##name(), name##_);                                              \
  }

#define WRITE_ENUM_FIELD(stream, name)                                                         \
  if (has_##name()) {                                                                          \
    write_enum_field(stream, order_##name(), name##_);                                         \
  }

#define WRITE_REPEATED_FIELD(stream, name)                                                     \
  if (has_##name()) {                                                                          \
    write_repeated_field(stream, order_##name(), name##_, packed_##name());                    \
  }

#define READ_BEGIN(stream, cls)                                                                \
  DEBUG_PRINT("+ read begin " #cls)                                                            \
  while (stream.not_end()) {                                                                   \
    utils::FieldNumber field_number = stream.next_field();                                     \
    DEBUG_PRINT2("  = field number ", field_number.string().c_str())                           \
    if (field_number.field_number == 0) {                                                      \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);        \
    }

#define READ_END(stream, cls)                                                                  \
  else {                                                                                       \
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);     \
  }                                                                                            \
  }                                                                                            \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(stream, name)                                                               \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    DEBUG_PRINT("  + field " #name)                                                            \
    read_field(stream, field_number.wire_type, name##_, #name);                                \
    DEBUG_PRINT("  - field " #name)                                                            \
  }

#define READ_ENUM_FIELD(stream, name)                                                          \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    DEBUG_PRINT("  + enum " #name)                                                             \
    read_enum_field(stream, field_number.wire_type, name##_, #name);                           \
    DEBUG_PRINT("  - enum " #name)                                                             \
  }

#define READ_REPEATED_FIELD(stream, name)                                                      \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    DEBUG_PRINT("  + repeat " #name)                                                           \
    read_repeated_field(stream, field_number.wire_type, name##_, #name, packed_##name());      \
    DEBUG_PRINT("  - repeat " #name)                                                           \
  }

namespace onnx2 {

////////
// write
////////

template <typename T>
void write_field(utils::BinaryWriteStream &stream, int order, const T &field) {
  utils::StringWriteStream local;
  field.SerializeToStream(local);
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string_stream(local);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const std::string &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const std::optional<uint64_t> &field) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(*field);
  }
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

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const int64_t &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_int64(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const uint64_t &field) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const int32_t &field) {
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
void write_field(utils::BinaryWriteStream &stream, int order,
                 const std::vector<uint8_t> &field) {
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
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const std::vector<T> &field, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (auto d : field) {
    utils::StringWriteStream local;
    d.SerializeToStream(local);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const std::vector<std::string> &field, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (auto d : field) {
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string(d);
  }
}

#define WRITE_REPEATED_FIELD_IMPL(type, unpack_method)                                         \
  template <>                                                                                  \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                       \
                            const std::vector<type> &field, bool is_packed) {                  \
    if (is_packed) {                                                                           \
      stream.write_field_header(order, FIELD_FIXED_SIZE);                                      \
      stream.write_variant_uint64(field.size() * sizeof(type));                                \
      for (auto d : field) {                                                                   \
        stream.write_packed_element(d);                                                        \
      }                                                                                        \
    } else {                                                                                   \
      for (auto d : field) {                                                                   \
        stream.write_field_header(order, FIELD_FIXED_SIZE);                                    \
        stream.unpack_method(d);                                                               \
      }                                                                                        \
    }                                                                                          \
  }

WRITE_REPEATED_FIELD_IMPL(double, write_double)
WRITE_REPEATED_FIELD_IMPL(float, write_float)

#define WRITE_REPEATED_FIELD_IMPL_INT(type)                                                    \
  template <>                                                                                  \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                       \
                            const std::vector<type> &field, bool is_packed) {                  \
    if (is_packed) {                                                                           \
      stream.write_field_header(order, FIELD_FIXED_SIZE);                                      \
      stream.write_variant_uint64(field.size());                                               \
      for (auto d : field) {                                                                   \
        stream.write_variant_uint64(static_cast<uint64_t>(d));                                 \
      }                                                                                        \
    } else {                                                                                   \
      for (auto d : field) {                                                                   \
        stream.write_field_header(order, FIELD_VARINT);                                        \
        stream.write_variant_uint64(static_cast<uint64_t>(d));                                 \
      }                                                                                        \
    }                                                                                          \
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
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  field.ParseFromStream(dim_buf);
}

template <>
void read_field<std::string>(utils::BinaryStream &stream, int wire_type, std::string &field,
                             const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::optional<uint64_t> &field,
                const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type,
                utils::OptionalField<uint64_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type,
                utils::OptionalField<int64_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type,
                utils::OptionalField<int32_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, uint64_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int64_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int32_t &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, float &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, double &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_double();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  uint64_t len = stream.next_uint64();
  field.resize(len);
  memcpy(field.data(), stream.read_bytes(len), len);
}

template <typename T>
void read_enum_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = static_cast<TensorProto::DataType>(stream.next_uint64());
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  T elem;
  elem.ParseFromStream(dim_buf);
  field.emplace_back(elem);
}

template <>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         std::vector<std::string> &field, const char *name, bool is_packed) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field.push_back(stream.next_string());
}

#define READ_REPEATED_FIELD_IMPL(type, unpack_method, unpacked_wire_type)                      \
  template <>                                                                                  \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type,                         \
                           std::vector<type> &field, const char *name, bool is_packed) {       \
    if (is_packed) {                                                                           \
      DEBUG_PRINT2("    read packed", name);                                                   \
      EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type,           \
                  " for field '", name, "'");                                                  \
      uint64_t size = stream.next_uint64();                                                    \
      EXT_ENFORCE(size % sizeof(type) == 0, "unexpected size ", size,                          \
                  ", it is not a multiple of sizeof(" #type ")");                              \
      size /= sizeof(type);                                                                    \
      field.resize(size);                                                                      \
      for (size_t i = 0; i < static_cast<size_t>(size); ++i) {                                 \
        stream.next_packed_element(field[i]);                                                  \
      }                                                                                        \
    } else {                                                                                   \
      DEBUG_PRINT2("    read unpacked", name);                                                 \
      EXT_ENFORCE(wire_type == unpacked_wire_type, "unexpected wire_type=", wire_type,         \
                  " for field '", name, "'");                                                  \
      field.push_back(stream.unpack_method());                                                 \
    }                                                                                          \
  }

READ_REPEATED_FIELD_IMPL(double, next_double, FIELD_FIXED_SIZE)
READ_REPEATED_FIELD_IMPL(float, next_float, FIELD_FIXED_SIZE)

#define READ_REPEATED_FIELD_IMPL_INT(type, unpack_method)                                      \
  template <>                                                                                  \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type,                         \
                           std::vector<type> &field, const char *name, bool is_packed) {       \
    if (is_packed) {                                                                           \
      DEBUG_PRINT2("    read packed", name);                                                   \
      EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type,           \
                  " for field '", name, "'");                                                  \
      uint64_t size = stream.next_uint64();                                                    \
      field.resize(size);                                                                      \
      for (size_t i = 0; i < static_cast<size_t>(size); ++i) {                                 \
        field[i] = stream.unpack_method();                                                     \
      }                                                                                        \
    } else {                                                                                   \
      DEBUG_PRINT2("    read unpacked", name);                                                 \
      EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type,               \
                  " for field '", name, "'");                                                  \
      field.push_back(stream.unpack_method());                                                 \
    }                                                                                          \
  }

READ_REPEATED_FIELD_IMPL_INT(int64_t, next_int64)
READ_REPEATED_FIELD_IMPL_INT(int32_t, next_int32)
READ_REPEATED_FIELD_IMPL_INT(uint64_t, next_uint64)

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         utils::RepeatedField<T> &field, const char *name, bool is_packed) {
  read_repeated_field(stream, wire_type, field.values, name, is_packed);
}

} // namespace onnx2
