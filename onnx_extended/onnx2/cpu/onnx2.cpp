#include "onnx2.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace validation {
namespace onnx2 {

void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream) {
  key.clear();
  value.clear();
  int n_read = 0;
  while (stream.not_end() && n_read < 2) {
    auto f = stream.next_field();
    EXT_ENFORCE(
        f.wire_type == FIELD_FIXED_SIZE,
        "[StringStringEntryProto::ParseFromStream] expected length-delimited wire type, field=",
        f.string());
    if (f.field_number == 1) {
      key = stream.next_string();
      ++n_read;
    } else if (f.field_number == 2) {
      value = stream.next_string();
      ++n_read;
    } else {
      EXT_THROW("[StringStringEntryProto::ParseFromStream] unknown field ", f.string());
    }
  }
}

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  if (!key.empty()) {
    stream.write_field_header(1, FIELD_FIXED_SIZE);
    stream.write_string(key);
  }
  if (!value.empty()) {
    stream.write_field_header(2, FIELD_FIXED_SIZE);
    stream.write_string(value);
  }
}

void TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream) {
  while (stream.not_end()) {
    auto f = stream.next_field();
    if (f.field_number == 1 && f.wire_type == FIELD_VARINT) {
      dim_value = stream.next_uint64();
    } else if (f.field_number == 2 && f.wire_type == FIELD_FIXED_SIZE) {
      dim_param = stream.next_string();
    } else if (f.field_number == 3 && f.wire_type == FIELD_FIXED_SIZE) {
      denotation = stream.next_string();
    } else {
      EXT_THROW("[TensorShapeProto::Dimension::ParseFromStream] unknown field ", f.string());
    }
  }
}

void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream) const {
  stream.write_field_header(1, FIELD_VARINT);
  stream.write_variant_uint64(dim_value);

  if (!dim_param.empty()) {
    stream.write_field_header(2, FIELD_FIXED_SIZE);
    stream.write_string(dim_param);
  }
  if (!denotation.empty()) {
    stream.write_field_header(3, FIELD_FIXED_SIZE);
    stream.write_string(denotation);
  }
}

void TensorShapeProto::ParseFromStream(utils::BinaryStream &stream) {
  while (stream.not_end()) {
    auto f = stream.next_field();
    if (f.field_number == 1 && f.wire_type == FIELD_FIXED_SIZE) { // repeated dim
      utils::StringStream dim_buf;
      stream.read_string_stream(dim_buf);
      Dimension d;
      d.ParseFromStream(dim_buf);
      dim.emplace_back(d);
    } else {
      EXT_THROW("[TensorShapeProto::ParseFromStream] unknown field n", f.string());
    }
  }
}

void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  for (auto d : dim) {
    utils::StringWriteStream local;
    d.SerializeToStream(local);
    stream.write_field_header(1, FIELD_FIXED_SIZE);
    stream.write_string_stream(local);
  }
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
      len = stream.next_uint64();
      stream.can_read(len, "[TensorProto::ParseFromStream] metadata_props");
      StringStringEntryProto entry;
      entry.ParseFromStream(stream);
      metadata_props.emplace_back(entry);
      break;
    }

    default:
      EXT_THROW("[TensorProto::ParseFromStream] unexpected field ", f.string());
    }
  }
}

} // namespace onnx2
} // namespace validation
