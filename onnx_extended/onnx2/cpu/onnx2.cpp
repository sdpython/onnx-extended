#include "onnx2.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

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
    write_repeated_field(stream, order_##name(), name##_);                                     \
  }

#define READ_BEGIN(stream, cls)                                                                \
  while (stream.not_end()) {                                                                   \
    utils::FieldNumber field_number = stream.next_field();                                     \
    if (field_number.field_number == 0) {                                                      \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);        \
    }

#define READ_END(stream, cls)                                                                  \
  else {                                                                                       \
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);     \
  }                                                                                            \
  }

#define READ_FIELD(stream, name)                                                               \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    read_field(stream, field_number.wire_type, name##_, #name);                                \
  }

#define READ_ENUM_FIELD(stream, name)                                                          \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    read_enum_field(stream, field_number.wire_type, name##_, #name);                           \
  }

#define READ_REPEATED_FIELD(stream, name)                                                      \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                    \
    read_repeated_field(stream, field_number.wire_type, name##_, #name);                       \
  }

namespace validation {
namespace onnx2 {

template <typename T>
void write_field(utils::BinaryWriteStream &stream, int order, const T &field);

template <>
void write_field<std::string>(utils::BinaryWriteStream &stream, int order,
                              const std::string &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  stream.write_string(field);
}

template <>
void write_field<std::optional<uint64_t>>(utils::BinaryWriteStream &stream, int order,
                                          const std::optional<uint64_t> &field) {
  if (field.has_value()) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(*field);
  }
}

template <>
void write_field<std::vector<uint8_t>>(utils::BinaryWriteStream &stream, int order,
                                       const std::vector<uint8_t> &field) {
  stream.write_field_header(order, FIELD_FIXED_SIZE);
  utils::BorrowedWriteStream local(field.data(), field.size());
  stream.write_string_stream(local);
}

template <typename T>
void write_repeated_field(utils::BinaryWriteStream &stream, int order,
                          const std::vector<T> &field) {
  for (auto d : field) {
    utils::StringWriteStream local;
    d.SerializeToStream(local);
    stream.write_field_header(order, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
}

template <typename T>
void write_enum_field(utils::BinaryWriteStream &stream, int order, const T &field) {
  stream.write_field_header(2, FIELD_VARINT);
  stream.write_variant_uint64(static_cast<uint64_t>(field));
}

template <>
void write_repeated_field<uint64_t>(utils::BinaryWriteStream &stream, int order,
                                    const std::vector<uint64_t> &field) {
  for (auto d : field) {
    stream.write_field_header(order, FIELD_VARINT);
    stream.write_variant_uint64(d);
  }
}

template <typename T>
void read_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name);

template <>
void read_field<std::string>(utils::BinaryStream &stream, int wire_type, std::string &field,
                             const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_string();
}

template <>
void read_field<std::optional<uint64_t>>(utils::BinaryStream &stream, int wire_type,
                                         std::optional<uint64_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = stream.next_uint64();
}

template <>
void read_field<std::vector<uint8_t>>(utils::BinaryStream &stream, int wire_type,
                                      std::vector<uint8_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  uint64_t len = stream.next_uint64();
  field.resize(len);
  memcpy(field.data(), stream.read_bytes(len), len);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name) {
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
                         std::vector<uint64_t> &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field.push_back(static_cast<int64_t>(stream.next_uint64()));
}

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_FIELD(stream, value)
}

template <typename T>
void read_enum_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type, " for field '",
              name, "'");
  field = static_cast<TensorProto::DataType>(stream.next_uint64());
}

void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, StringStringEntryProto)
  READ_FIELD(stream, key)
  READ_FIELD(stream, value)
  READ_END(stream, StringStringEntryProto)
}

void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, dim_value)
  WRITE_FIELD(stream, dim_param)
  WRITE_FIELD(stream, denotation)
}

void TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, StringStringEntryProto)
  READ_FIELD(stream, dim_value)
  READ_FIELD(stream, dim_param)
  READ_FIELD(stream, denotation)
  READ_END(stream, StringStringEntryProto)
}

void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dim)
}

void TensorShapeProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, StringStringEntryProto)
  READ_REPEATED_FIELD(stream, dim)
  READ_END(stream, StringStringEntryProto)
}

void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dims)
  WRITE_ENUM_FIELD(stream, data_type)
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, raw_data)
  WRITE_FIELD(stream, doc_string)
  WRITE_REPEATED_FIELD(stream, external_data)
  WRITE_REPEATED_FIELD(stream, metadata_props)
}

void TensorProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, StringStringEntryProto)
  READ_REPEATED_FIELD(stream, dims)
  READ_ENUM_FIELD(stream, data_type)
  READ_FIELD(stream, name)
  READ_FIELD(stream, doc_string)
  READ_FIELD(stream, raw_data)
  READ_REPEATED_FIELD(stream, external_data)
  READ_REPEATED_FIELD(stream, metadata_props)
  READ_END(stream, StringStringEntryProto)
}

} // namespace onnx2
} // namespace validation
