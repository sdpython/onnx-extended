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

////////////////
// macro helpers
////////////////

#define NAME_EXIST_VALUE(name) name_exist_value(_name_##name, has_##name(), ptr_##name())

#define IMPLEMENT_PROTO(cls)                                                                           \
  void cls::CopyFrom(const cls &proto) {                                                               \
    utils::StringWriteStream stream;                                                                   \
    SerializeOptions opts;                                                                             \
    proto.SerializeToStream(stream, opts);                                                             \
    utils::StringStream read_stream(stream.data(), stream.size());                                     \
    ParseOptions ropts;                                                                                \
    ParseFromStream(read_stream, ropts);                                                               \
  }                                                                                                    \
  uint64_t cls::SerializeSize() const {                                                                \
    SerializeOptions opts;                                                                             \
    utils::StringWriteStream stream;                                                                   \
    return SerializeSize(stream, opts);                                                                \
  }                                                                                                    \
  void cls::ParseFromString(const std::string &raw) {                                                  \
    ParseOptions opts;                                                                                 \
    ParseFromString(raw, opts);                                                                        \
  }                                                                                                    \
  void cls::ParseFromString(const std::string &raw, ParseOptions &opts) {                              \
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());                                \
    onnx2::utils::StringStream st(ptr, raw.size());                                                    \
    ParseFromStream(st, opts);                                                                         \
  }                                                                                                    \
  void cls::SerializeToString(std::string &out) const {                                                \
    SerializeOptions opts;                                                                             \
    SerializeToString(out, opts);                                                                      \
  }                                                                                                    \
  void cls::SerializeToString(std::string &out, SerializeOptions &opts) const {                        \
    onnx2::utils::StringWriteStream buf;                                                               \
    auto &opts_ref = opts;                                                                             \
    SerializeToStream(buf, opts_ref);                                                                  \
    out = std::string(reinterpret_cast<const char *>(buf.data()), buf.size());                         \
  }

///////////////////////
// macro serialize size
///////////////////////

#define SIZE_FIELD(size, options, stream, name)                                                        \
  if (has_##name()) {                                                                                  \
    size += size_field(stream, order_##name(), ref_##name(), options);                                 \
  }

#define SIZE_FIELD_LIMIT(size, options, stream, name)                                                  \
  if (has_##name()) {                                                                                  \
    size += size_field_limit(stream, order_##name(), ref_##name(), options);                           \
  }

#define SIZE_ENUM_FIELD(size, options, stream, name)                                                   \
  if (has_##name()) {                                                                                  \
    size += size_enum_field(stream, order_##name(), ref_##name(), options);                            \
  }

#define SIZE_REPEATED_FIELD(size, options, stream, name)                                               \
  if (has_##name()) {                                                                                  \
    size += size_repeated_field(stream, order_##name(), name##_, packed_##name(), options);            \
  }

#define SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, name)                                         \
  if (has_##name()) {                                                                                  \
    size += size_optional_proto_field(stream, order_##name(), name##_optional(), options);             \
  }

//////////////
// macro write
//////////////

#define WRITE_FIELD(options, stream, name)                                                             \
  if (has_##name()) {                                                                                  \
    write_field(stream, order_##name(), ref_##name(), options);                                        \
  }

#define WRITE_FIELD_NULL(options, stream, name)                                                        \
  if (!name##_.null()) {                                                                               \
    write_field(stream, order_##name(), ref_##name(), options);                                        \
  }

#define WRITE_FIELD_EMPTY(options, stream, name)                                                       \
  write_field(stream, order_##name(), ref_##name(), options);

#define WRITE_FIELD_LIMIT(options, stream, name)                                                       \
  if (has_##name()) {                                                                                  \
    write_field_limit(stream, order_##name(), ref_##name(), options);                                  \
  }

#define WRITE_ENUM_FIELD(options, stream, name)                                                        \
  if (has_##name()) {                                                                                  \
    write_enum_field(stream, order_##name(), ref_##name(), options);                                   \
  }

#define WRITE_REPEATED_FIELD(options, stream, name)                                                    \
  if (has_##name()) {                                                                                  \
    write_repeated_field(stream, order_##name(), name##_, packed_##name(), options);                   \
  }

#define WRITE_OPTIONAL_PROTO_FIELD(options, stream, name)                                              \
  if (has_##name()) {                                                                                  \
    write_optional_proto_field(stream, order_##name(), name##_optional(), options);                    \
  }

/////////////
// macro read
/////////////

#define READ_BEGIN(options, stream, cls)                                                               \
  DEBUG_PRINT("+ read begin " #cls)                                                                    \
  while (stream.not_end()) {                                                                           \
    utils::FieldNumber field_number = stream.next_field();                                             \
    DEBUG_PRINT2("  = field number ", field_number.string().c_str())                                   \
    if (field_number.field_number == 0) {                                                              \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);                \
    }

#define READ_END(options, stream, cls)                                                                 \
  else {                                                                                               \
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);             \
  }                                                                                                    \
  }                                                                                                    \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(options, stream, name)                                                              \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + field " #name)                                                                    \
    read_field(stream, field_number.wire_type, name##_, #name, options);                               \
    DEBUG_PRINT("  - field " #name)                                                                    \
  }

#define READ_FIELD_LIMIT(options, stream, name)                                                        \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + field " #name)                                                                    \
    read_field_limit(stream, field_number.wire_type, name##_, #name, options);                         \
    DEBUG_PRINT("  - field " #name)                                                                    \
  }

#define READ_OPTIONAL_PROTO_FIELD(options, stream, name)                                               \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + optional field " #name)                                                           \
    read_optional_proto_field(stream, field_number.wire_type, name##_, #name, options);                \
    DEBUG_PRINT("  - optional field " #name)                                                           \
  }

#define READ_ENUM_FIELD(options, stream, name)                                                         \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + enum " #name)                                                                     \
    read_enum_field(stream, field_number.wire_type, name##_, #name, options);                          \
    DEBUG_PRINT("  - enum " #name)                                                                     \
  }

#define READ_REPEATED_FIELD(options, stream, name)                                                     \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + repeat " #name)                                                                   \
    read_repeated_field(stream, field_number.wire_type, name##_, #name, packed_##name(), options);     \
    DEBUG_PRINT("  - repeat " #name)                                                                   \
  }

using namespace onnx_extended_helpers;

namespace onnx2 {

////////
// serialized size
////////

template <typename T>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const T &field,
                    SerializeOptions &options) {
  auto s = field.SerializeSize(stream, options);
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
}

template <typename T>
uint64_t size_optional_proto_field(utils::BinaryWriteStream &stream, int order,
                                   const utils::OptionalField<T> &field, SerializeOptions &options) {
  if (field.has_value()) {
    auto s = (*field).SerializeSize(stream, options);
    return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const utils::String &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_string(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order,
                    const utils::OptionalField<uint64_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_variant_uint64(*field);
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order,
                    const utils::OptionalField<int64_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_int64(*field);
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order,
                    const utils::OptionalField<int32_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_int32(*field);
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const int64_t &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_int64(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const uint64_t &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_variant_uint64(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const int32_t &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_int32(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const double &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_double(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const float &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_float(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream &stream, int order, const std::vector<uint8_t> &field,
                    SerializeOptions &) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(field.size()) +
         field.size();
}

uint64_t size_field_limit(utils::BinaryWriteStream &stream, int order,
                          const std::vector<uint8_t> &field, SerializeOptions &options) {
  if (!options.skip_raw_data || field.size() < static_cast<size_t>(options.raw_data_threshold)) {
    return size_field(stream, order, field, options);
  }
  return 0;
}

template <typename T>
uint64_t size_enum_field(utils::BinaryWriteStream &stream, int order, const T &field,
                         SerializeOptions &) {
  return stream.size_field_header(order, FIELD_VARINT) +
         stream.VarintSize(static_cast<uint64_t>(field));
}

template <typename T>
uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order, const std::vector<T> &field,
                             bool is_packed, SerializeOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (const auto &d : field) {
    auto s = d.SerializeSize(stream, options);
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return size;
}

template <>
uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order,
                             const std::vector<utils::String> &field, bool is_packed,
                             SerializeOptions &) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (const auto &d : field) {
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(d.size()) + d.size();
  }
  return size;
}

#define SIZE_REPEATED_FIELD_IMPL(type, unpack_method)                                                  \
  template <>                                                                                          \
  uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order,                            \
                               const std::vector<type> &field, bool is_packed, SerializeOptions &) {   \
    if (is_packed) {                                                                                   \
      return stream.size_field_header(order, FIELD_FIXED_SIZE) +                                       \
             stream.VarintSize(field.size() * sizeof(type)) + field.size() * sizeof(type);             \
    } else {                                                                                           \
      uint64_t size = 0;                                                                               \
      for (const auto &d : field) {                                                                    \
        size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.unpack_method(d);           \
      }                                                                                                \
      return size;                                                                                     \
    }                                                                                                  \
  }

SIZE_REPEATED_FIELD_IMPL(double, size_double)
SIZE_REPEATED_FIELD_IMPL(float, size_float)

#define SIZE_REPEATED_FIELD_IMPL_INT(type)                                                             \
  template <>                                                                                          \
  uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order,                            \
                               const std::vector<type> &field, bool is_packed, SerializeOptions &) {   \
    if (is_packed) {                                                                                   \
      uint64_t size =                                                                                  \
          stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(field.size());         \
      for (const auto &d : field) {                                                                    \
        size += stream.VarintSize(static_cast<uint64_t>(d));                                           \
      }                                                                                                \
      return size;                                                                                     \
    } else {                                                                                           \
      uint64_t size = 0;                                                                               \
      for (const auto &d : field) {                                                                    \
        size += stream.size_field_header(order, FIELD_VARINT);                                         \
        size += stream.VarintSize(static_cast<uint64_t>(d));                                           \
      }                                                                                                \
      return size;                                                                                     \
    }                                                                                                  \
  }

SIZE_REPEATED_FIELD_IMPL_INT(uint64_t)
SIZE_REPEATED_FIELD_IMPL_INT(int64_t)
SIZE_REPEATED_FIELD_IMPL_INT(int32_t)

template <typename T>
uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order,
                             const utils::RepeatedField<T> &field, bool is_packed,
                             SerializeOptions &options) {
  return size_repeated_field(stream, order, field.values(), is_packed, options);
}

template <typename T>
uint64_t size_repeated_field(utils::BinaryWriteStream &stream, int order,
                             const utils::RepeatedProtoField<T> &field, bool is_packed,
                             SerializeOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (size_t i = 0; i < field.size(); ++i) {
    auto s = field[i].SerializeSize(stream, options);
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return size;
}

////////
// write
////////

template <typename T>
void write_field(utils::BinaryWriteStream &stream, int order, const T &field,
                 SerializeOptions &options) {
  // TODO: avoid copy
  // If we could know the size of the field in advance (after it is serialized),
  // we could avoid to serialize into a buffer and then write it (copy it in face)
  // to the stream.
  utils::StringWriteStream local;
  field.SerializeToStream(local, options);
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string_stream(local);
}

template <typename T>
void write_optional_proto_field(utils::BinaryWriteStream &stream, int order,
                                const utils::OptionalField<T> &field, SerializeOptions &options) {
  if (field.has_value()) {
    // TODO: avoid copy
    // If we could know the size of the field in advance (after it is serialized),
    // we could avoid to serialize into a buffer and then write it (copy it in face)
    // to the stream.
    utils::StringWriteStream local;
    (*field).SerializeToStream(local, options);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const utils::String &field,
                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<uint64_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(*field);
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<int64_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(static_cast<uint64_t>(*field));
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order,
                 const utils::OptionalField<int32_t> &field, SerializeOptions &) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(static_cast<uint64_t>(*field));
  }
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const int64_t &field,
                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_int64(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const uint64_t &field,
                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const int32_t &field,
                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_int32(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const double &field, SerializeOptions &) {
  stream.write_field_header(order, FIELD_FIXED_SIZE); // FIELD_FIXED64);
  stream.write_double(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const float &field, SerializeOptions &) {
  stream.write_field_header(order, FIELD_FIXED_SIZE); // FIELD_FIXED32);
  stream.write_float(field);
}

template <>
void write_field(utils::BinaryWriteStream &stream, int order, const std::vector<uint8_t> &field,
                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  utils::BorrowedWriteStream local(field.data(), field.size());
  stream.write_string_stream(local);
}

void write_field_limit(utils::BinaryWriteStream &stream, int order, const std::vector<uint8_t> &field,
                       SerializeOptions &options) {
  if (!options.skip_raw_data || field.size() < static_cast<size_t>(options.raw_data_threshold)) {
    write_field(stream, order, field, options);
  }
}

template <typename T>
void write_enum_field(utils::BinaryWriteStream &stream, int order, const T &field, SerializeOptions &) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(static_cast<uint64_t>(field));
}

template <typename T>
void write_repeated_field(utils::BinaryWriteStream &stream, int order, const std::vector<T> &field,
                          bool is_packed, SerializeOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (const auto &d : field) {
    // TODO: avoid copy
    // If we could know the size of the field in advance (after it is serialized),
    // we could avoid to serialize into a buffer and then write it (copy it in face)
    // to the stream.
    utils::StringWriteStream local;
    d.SerializeToStream(local, options);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const std::vector<utils::String> &field, bool is_packed, SerializeOptions &) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (const auto &d : field) {
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string(d);
  }
}

#define WRITE_REPEATED_FIELD_IMPL(type, unpack_method)                                                 \
  template <>                                                                                          \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                               \
                            const std::vector<type> &field, bool is_packed, SerializeOptions &) {      \
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
                            const std::vector<type> &field, bool is_packed, SerializeOptions &) {      \
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
                          const utils::RepeatedField<T> &field, bool is_packed,
                          SerializeOptions &options) {
  write_repeated_field(stream, order, field.values(), is_packed, options);
}

template <typename T>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const utils::RepeatedProtoField<T> &field, bool is_packed,
                          SerializeOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  for (size_t i = 0; i < field.size(); ++i) {
    // TODO: avoid copy
    // If we could know the size of the field in advance (after it is serialized),
    // we could avoid to serialize into a buffer and then write it (copy it in face)
    // to the stream.
    utils::StringWriteStream local;
    field[i].SerializeToStream(local, options);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

///////
// read
///////

template <typename T>
void read_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name,
                ParseOptions &options) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
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
              "'");
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
              "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::String &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int64_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int32_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<float> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, uint64_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int64_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int32_t &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, float &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, double &field, const char *name,
                ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  field = stream.next_double();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                const char *name, ParseOptions &) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
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
                "'");
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
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '", name, "'");
  field = static_cast<T>(stream.next_uint64());
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name, bool is_packed, ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
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
              "'");
  field.emplace_back(utils::String(stream.next_string()));
}

#define READ_REPEATED_FIELD_IMPL(type, unpack_method, unpacked_wire_type)                              \
  template <>                                                                                          \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<type> &field,       \
                           const char *name, bool is_packed, ParseOptions &) {                         \
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
                           const char *name, bool is_packed, ParseOptions &) {                         \
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
                         const char *name, bool is_packed, ParseOptions &options) {
  read_repeated_field(stream, wire_type, field.mutable_values(), name, is_packed, options);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         utils::RepeatedProtoField<T> &field, const char *name, bool is_packed,
                         ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '", name,
              "'");
  utils::StringStream dim_buf;
  stream.read_string_stream(dim_buf);
  T &elem = field.add();
  elem.ParseFromStream(dim_buf, options);
  stream.set_lock(false);
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

template <typename T> std::string write_as_string(utils::PrintOptions &, const T &field) {
  return MakeString(field);
}

template <> std::string write_as_string(utils::PrintOptions &, const utils::String &field) {
  return MakeString("\"", field.as_string(), "\"");
}

template <> std::string write_as_string(utils::PrintOptions &, const std::vector<uint8_t> &field) {
  const char *hex_chars = "0123456789ABCDEF";
  std::stringstream result;
  for (const auto &b : field) {
    result << hex_chars[b / 16] << hex_chars[b % 16];
  }
  return result.str();
}

template <typename T>
std::string write_as_string_vector(utils::PrintOptions &, const std::vector<T> &field) {
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

template <typename T>
std::string write_as_repeated_field(utils::PrintOptions &, const utils::RepeatedField<T> &field) {
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

template <>
std::string write_as_repeated_field(utils::PrintOptions &,
                                    const utils::RepeatedField<utils::String> &field) {
  std::stringstream result;
  result << "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result << field[i].as_string();
    if (i + 1 != field.size())
      result << ", ";
  }
  result << "]";
  return result.str();
}

template <typename T>
std::string write_as_string_optional(utils::PrintOptions &options, const std::optional<T> &field) {
  if (!field)
    return "null";
  return write_as_string(options, *field);
}

template <> std::string write_as_string(utils::PrintOptions &options, const std::vector<float> &field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::vector<int64_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::vector<uint64_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::vector<double> &field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::vector<int32_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::optional<float> &field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::optional<int64_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::optional<uint64_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::optional<double> &field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const std::optional<int32_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const utils::RepeatedField<float> &field) {
  return write_as_repeated_field(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options, const utils::RepeatedField<int64_t> &field) {
  return write_as_repeated_field(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions &options,
                            const utils::RepeatedField<utils::String> &field) {
  return write_as_repeated_field(options, field);
}

template <typename... Args>
std::string write_as_string(utils::PrintOptions &options, const Args &...args) {
  std::stringstream result;
  result << "{";

  auto append_arg = [&options, &result, first = true](const auto &arg) mutable {
    if (arg.exist) {
      if (!first) {
        result << ", ";
      }
      first = false;
      result << arg.name;
      result << ": ";
      result << write_as_string(options, *arg.value);
    }
  };

  (append_arg(args), ...);
  result << "}";
  return result.str();
}

template <typename T>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const T &field) {
  std::vector<std::string> r = field.PrintToVectorString(options);
  if (r.size() <= 1) {
    return {MakeString(field_name, ": ", r.back(), ",")};
  } else {
    std::vector<std::string> rows{MakeString(field_name, ": ")};
    for (size_t i = 0; i < r.size(); ++i) {
      if (i == 0) {
        rows[0] += r[0];
      } else if (i + 1 == r.size()) {
        rows.push_back(MakeString(r[i]));
      } else {
        rows.push_back(r[i]);
      }
    }
    return rows;
  }
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::String &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const int64_t &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const float &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const uint64_t &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const int32_t &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const TensorProto::DataType &field) {
  return {MakeString(field_name, ": ", write_as_string(options, static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const AttributeProto::AttributeType &field) {
  return {MakeString(field_name, ": ", write_as_string(options, static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const std::vector<uint8_t> &field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &, const char *field_name,
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
std::vector<std::string> write_into_vector_string_repeated(utils::PrintOptions &,
                                                           const char *field_name,
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
  return rows;
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::RepeatedField<uint64_t> &field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::RepeatedField<int64_t> &field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::RepeatedField<int32_t> &field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::RepeatedField<float> &field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::RepeatedField<double> &field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <typename T>
std::vector<std::string> write_into_vector_string_optional(utils::PrintOptions &options,
                                                           const char *field_name,
                                                           const utils::OptionalField<T> &field) {
  if (field.has_value()) {
    return {MakeString(field_name, ": ", write_as_string(options, *field), ",")};
  } else {
    return {MakeString(field_name, ": null,")};
  }
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::OptionalField<int64_t> &field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::OptionalField<uint64_t> &field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(utils::PrintOptions &options, const char *field_name,
                                                  const utils::OptionalField<int32_t> &field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <typename... Args>
std::vector<std::string> write_proto_into_vector_string(utils::PrintOptions &options,
                                                        const Args &...args) {
  std::vector<std::string> rows{"{"};
  auto append_arg = [&options, &rows, first = true](const auto &arg) mutable {
    if (arg.exist) {
      std::vector<std::string> r = write_into_vector_string(options, arg.name, *arg.value);
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
