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

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_FIELD(stream, value)
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

void TensorProto::ParseFromStream(utils::BinaryStream &stream) {
  uint64_t len;
  while (stream.not_end()) {
    auto f = stream.next_field();
    switch (f.field_number) {
    case 1: // dims (repeated int64, varint)
      EXT_ENFORCE(f.wire_type == FIELD_VARINT,
                  "[TensorProto::ParseFromStream] dims: wrong wire type (dims), field ",
                  f.string());
      dims.push_back(static_cast<int64_t>(stream.next_uint64()));
      break;

    case 2: // data_type (int32, varint)
      EXT_ENFORCE(f.wire_type == FIELD_VARINT,
                  "[TensorProto::ParseFromStream] data_type: wrong wire type, field ",
                  f.string());
      data_type = static_cast<TensorProto::DataType>(stream.next_uint64());
      break;

    case 8: // name
      EXT_ENFORCE(f.wire_type == FIELD_FIXED_SIZE,
                  "[TensorProto::ParseFromStream] name: wrong wire type, field ", f.string());
      name = stream.next_string();
      break;

    case 9: // raw_data (bytes)
      // Maybe we should avoid a copy here.
      EXT_ENFORCE(f.wire_type == FIELD_FIXED_SIZE,
                  "[TensorProto::ParseFromStream] raw_data: wrong wire type, field ",
                  f.string());
      len = stream.next_uint64();
      raw_data.resize(len);
      memcpy(raw_data.data(), stream.read_bytes(len), len);
      break;

    case 12: // doc_string
      EXT_ENFORCE(f.wire_type == FIELD_FIXED_SIZE,
                  "[TensorProto::ParseFromStream] doc_string: wrong wire type, field ",
                  f.string());
      doc_string = stream.next_string();
      break;

    case 16: { // metadata_props
      EXT_ENFORCE(f.wire_type == FIELD_FIXED_SIZE,
                  "[TensorProto::ParseFromStream] metadata_props: wrong wire type, field",
                  f.string());
      utils::StringStream dim_buf;
      stream.read_string_stream(dim_buf);
      StringStringEntryProto entry;
      entry.ParseFromStream(dim_buf);
      metadata_props.emplace_back(entry);
      break;
    }

    default:
      EXT_THROW("[TensorProto::ParseFromStream] unexpected field ", f.string());
    }
  }
}

void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  // dims
  for (auto d : dims) {
    stream.write_field_header(1, FIELD_VARINT);
    stream.write_variant_uint64(d);
  }
  // data_type
  stream.write_field_header(2, FIELD_VARINT);
  stream.write_variant_uint64(static_cast<uint64_t>(data_type));
  // name
  stream.write_field_header(8, FIELD_FIXED_SIZE);
  stream.write_string(name);
  // raw_data
  stream.write_field_header(9, FIELD_FIXED_SIZE);
  utils::BorrowedWriteStream local(raw_data.data(), raw_data.size());
  stream.write_string_stream(local);
  // doc_string
  stream.write_field_header(12, FIELD_FIXED_SIZE);
  stream.write_string(doc_string);
  // metadata_props
  for (auto entry : metadata_props) {
    stream.write_field_header(16, FIELD_FIXED_SIZE);
    utils::StringWriteStream lw;
    entry.SerializeToStream(lw);
    stream.write_string_stream(lw);
  }
}

} // namespace onnx2
} // namespace validation
