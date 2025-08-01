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

using namespace onnx_extended_helpers;

namespace onnx2 {

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
void write_field(utils::BinaryWriteStream &stream, int order, const utils::OptionalField<float> &field,
                 SerializeOptions &) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(static_cast<float>(*field));
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

// repeated fields

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

// unpacked numbers

template <typename T>
void write_unpacked_number_float(utils::BinaryWriteStream &stream, int order, const T &value) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_packed_element(value);
}

template <typename T>
void write_unpacked_number_int(utils::BinaryWriteStream &stream, int order, const T &value) {
  stream.write_field_header(order, FIELD_VARINT);
  stream.write_variant_uint64(static_cast<uint64_t>(value));
}

template <typename T>
void write_unpacked_number(utils::BinaryWriteStream &stream, int order, const T &value);

#define WRITE_UNPACKED_NUMBER_FLOAT(type)                                                              \
  template <>                                                                                          \
  void write_unpacked_number(utils::BinaryWriteStream &stream, int order, const type &value) {         \
    write_unpacked_number_float(stream, order, value);                                                 \
  }

WRITE_UNPACKED_NUMBER_FLOAT(float)
WRITE_UNPACKED_NUMBER_FLOAT(double)

#define WRITE_UNPACKED_NUMBER_INT(type)                                                                \
  template <>                                                                                          \
  void write_unpacked_number(utils::BinaryWriteStream &stream, int order, const type &value) {         \
    write_unpacked_number_int(stream, order, value);                                                   \
  }

WRITE_UNPACKED_NUMBER_INT(uint64_t)
WRITE_UNPACKED_NUMBER_INT(int64_t)
WRITE_UNPACKED_NUMBER_INT(int32_t)

// packed numbers

template <typename T>
void write_repeated_field_packed_numerical_float(utils::BinaryWriteStream &stream, int order,
                                                 const std::vector<T> &field, bool,
                                                 SerializeOptions &) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_variant_uint64(field.size() * sizeof(T));
  for (const T &d : field) {
    stream.write_packed_element(d);
  }
}

template <typename T>
void write_repeated_field_packed_numerical_int(utils::BinaryWriteStream &stream, int order,
                                               const std::vector<T> &field, bool, SerializeOptions &) {
  utils::StringWriteStream local;
  for (const T &d : field) {
    local.write_variant_uint64(static_cast<uint64_t>(d));
  }
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string_stream(utils::BorrowedWriteStream(stream.data(), local.size()));
}

template <typename T>
void write_repeated_field_packed_numerical(utils::BinaryWriteStream &stream, int order,
                                           const std::vector<T> &field, bool, SerializeOptions &);

#define WRITE_PACKED_NUMBER_REPEATED_FLOAT(type)                                                       \
  template <>                                                                                          \
  void write_repeated_field_packed_numerical(utils::BinaryWriteStream &stream, int order,              \
                                             const std::vector<type> &field, bool is_packed,           \
                                             SerializeOptions &options) {                              \
    write_repeated_field_packed_numerical_float(stream, order, field, is_packed, options);             \
  }

WRITE_PACKED_NUMBER_REPEATED_FLOAT(float)
WRITE_PACKED_NUMBER_REPEATED_FLOAT(double)

#define WRITE_PACKED_NUMBER_REPEATED_INT(type)                                                         \
  template <>                                                                                          \
  void write_repeated_field_packed_numerical(utils::BinaryWriteStream &stream, int order,              \
                                             const std::vector<type> &field, bool is_packed,           \
                                             SerializeOptions &options) {                              \
    write_repeated_field_packed_numerical_int(stream, order, field, is_packed, options);               \
  }

WRITE_PACKED_NUMBER_REPEATED_INT(int64_t)
WRITE_PACKED_NUMBER_REPEATED_INT(int32_t)
WRITE_PACKED_NUMBER_REPEATED_INT(uint64_t)

// main function to write repeated numerical numbers

template <typename T>
void write_repeated_field_numerical(utils::BinaryWriteStream &stream, int order,
                                    const std::vector<T> &field, bool is_packed,
                                    SerializeOptions &options) {
  if (is_packed) {
    return write_repeated_field_packed_numerical(stream, order, field, is_packed, options);
  } else {
    for (const T &d : field) {
      write_unpacked_number(stream, order, d);
    }
  }
}

#define WRITE_REPEATED_FIELD_IMPL(type)                                                                \
  template <>                                                                                          \
  void write_repeated_field(utils::BinaryWriteStream &stream, int order,                               \
                            const std::vector<type> &field, bool is_packed,                            \
                            SerializeOptions &options) {                                               \
    write_repeated_field_numerical(stream, order, field, is_packed, options);                          \
  }

WRITE_REPEATED_FIELD_IMPL(double)
WRITE_REPEATED_FIELD_IMPL(float)
WRITE_REPEATED_FIELD_IMPL(uint64_t)
WRITE_REPEATED_FIELD_IMPL(int64_t)
WRITE_REPEATED_FIELD_IMPL(int32_t)

} // namespace onnx2
