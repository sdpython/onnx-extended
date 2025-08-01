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

using namespace onnx_extended_helpers;

namespace onnx2 {

template <typename T>
void read_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name,
                ParseOptions &options) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  field.ParseFromStream(dim_buf, options);
  stream.set_lock(false);
}

template <typename T>
void read_optional_proto_field(utils::BinaryStream &stream, int wire_type,
                               utils::OptionalField<T> &field, const char *name,
                               ParseOptions &options) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  field.set_empty_value();
  (*field).ParseFromStream(dim_buf, options);
  stream.set_lock(false);
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::RefString &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::String &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int64_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int32_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<float> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE || wire_type == FIELD_FIXED32,
              "unexpected wire_type=", wire_type, " for field '", name, "' at position '",
              stream.tell_around(), "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, uint64_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int64_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int32_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, float &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE || wire_type == FIELD_FIXED32,
              "unexpected wire_type=", wire_type, " for field '", name, "' at position '",
              stream.tell_around(), "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, double &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = stream.next_double();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  uint64_t len = stream.next_uint64();
  field.resize(len);
  memcpy(field.data(), stream.read_bytes(len), len);
}

void read_field_limit(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                      const char *name, ParseOptions &options) {
  if (!options.skip_raw_data) {
    read_field(stream, wire_type, field, name, options);
  } else {
    EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
                "' at position '", stream.tell_around(), "'");
    uint64_t len = stream.next_uint64();
    if (static_cast<int64_t>(len) < options.raw_data_threshold) {
      field.resize(len);
      memcpy(field.data(), stream.read_bytes(len), len);
    } else {
      stream.skip_bytes(len);
    }
  }
}

template <typename T>
void read_enum_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name,
                     ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field = static_cast<T>(stream.next_uint64());
}

// repeated fields

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, utils::RepeatedField<T> &field,
                         const char *name, bool is_packed, ParseOptions &options) {
  read_repeated_field(stream, wire_type, field.mutable_values(), name, is_packed, options);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         utils::RepeatedProtoField<T> &field, const char *name, bool is_packed,
                         ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name,
              "' at position '", stream.tell_around(), "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  T &elem = field.add();
  elem.ParseFromStream(dim_buf, options);
  stream.set_lock(false);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name, bool is_packed, ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  T elem;
  elem.ParseFromStream(dim_buf, options);
  field.emplace_back(elem);
  stream.set_lock(false);
}

template <>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<utils::String> &field,
                         const char *name, bool is_packed, ParseOptions &) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "' at position '", stream.tell_around(), "'");
  field.emplace_back(utils::String(stream.next_string()));
}

// numbers

template <typename T> T read_unpacked_number_float(utils::BinaryStream &stream, int wire_type) {
  // same as packed.
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type);
  T value;
  stream.next_packed_element(value);
  return value;
}

template <typename T> T read_unpacked_number_int(utils::BinaryStream &stream, int wire_type) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type);
  uint64_t i = stream.next_uint64();
  return static_cast<T>(i);
}

template <typename T> T read_unpacked_number(utils::BinaryStream &stream, int wire_type);

#define READ_UNPACKED_NUMBER_FLOAT(type)                                                               \
  template <> type read_unpacked_number(utils::BinaryStream &stream, int wire_type) {                  \
    return read_unpacked_number_float<type>(stream, wire_type);                                        \
  }

READ_UNPACKED_NUMBER_FLOAT(float)
READ_UNPACKED_NUMBER_FLOAT(double)

#define READ_UNPACKED_NUMBER_INT(type)                                                                 \
  template <> type read_unpacked_number(utils::BinaryStream &stream, int wire_type) {                  \
    return read_unpacked_number_int<type>(stream, wire_type);                                          \
  }

READ_UNPACKED_NUMBER_INT(uint64_t)
READ_UNPACKED_NUMBER_INT(int64_t)
READ_UNPACKED_NUMBER_INT(int32_t)

template <typename T>
void read_repeated_field_numerical(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                                   const char *name, bool is_packed, ParseOptions &) {
  if (is_packed) {
    DEBUG_PRINT2("    read packed", name);
    EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
                "' at position '", stream.tell_around(), "'");
    uint64_t size = stream.next_uint64();
    EXT_ENFORCE(size % sizeof(T) == 0, "unexpected size ", size, ", it is not a multiple of sizeof(",
                typeid(T).name(), ") for field '", name, "' at position '", stream.tell_around(), "'");
    size /= sizeof(T);
    field.resize(size);
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      stream.next_packed_element(field[i]);
    }
  } else {
    DEBUG_PRINT2("    read unpacked", name);
    field.push_back(read_unpacked_number<T>(stream, wire_type));
  }
}

#define READ_REPEATED_FIELD_IMPL(type)                                                                 \
  template <>                                                                                          \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<type> &field,       \
                           const char *name, bool is_packed, ParseOptions &options) {                  \
    read_repeated_field_numerical(stream, wire_type, field, name, is_packed, options);                 \
  }

READ_REPEATED_FIELD_IMPL(double)
READ_REPEATED_FIELD_IMPL(float)
READ_REPEATED_FIELD_IMPL(uint64_t)
READ_REPEATED_FIELD_IMPL(int64_t)
READ_REPEATED_FIELD_IMPL(int32_t)

} // namespace onnx2
