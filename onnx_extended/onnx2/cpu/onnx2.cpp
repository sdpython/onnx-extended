#include "onnx2.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace validation {
namespace onnx2 {

void StringStringEntryProto::ParseFromString(utils::BinaryStream &stream) {
  key.clear();
  value.clear();
  int n_read = 0;
  while (stream.not_end() && n_read < 2) {
    uint64_t field_key = stream.next_uint64();
    uint32_t field_number = field_key >> 3;
    uint32_t wire_type = field_key & 0x07;
    EXT_ENFORCE(
        wire_type == 2,
        "[StringStringEntryProto::ParseFromString] expected length-delimited wire type");
    if (field_number == 1) {
      key = stream.next_string();
      ++n_read;
    } else if (field_number == 2) {
      value = stream.next_string();
      ++n_read;
    } else {
      EXT_THROW("[StringStringEntryProto::ParseFromString] unknown field number: ",
                field_number);
    }
  }
}

void TensorShapeProto::Dimension::ParseFromString(utils::BinaryStream &stream) {
  while (stream.not_end()) {
    uint64_t key = stream.next_uint64();
    uint64_t field_number = key >> 3;
    uint64_t wire_type = key & 0x07;

    if (field_number == 1 && wire_type == 0) {
      dim_value = stream.next_uint64();
    } else if (field_number == 2 && wire_type == 2) {
      dim_param = stream.next_string();
    } else if (field_number == 3 && wire_type == 2) {
      denotation = stream.next_string();
    } else {
      EXT_THROW("[TensorShapeProto::Dimension::ParseFromString] unknown field number: ",
                field_number);
    }
  }
}

void TensorShapeProto::ParseFromString(utils::BinaryStream &stream) {
  while (stream.not_end()) {
    uint64_t key = stream.next_uint64();
    uint64_t field_number = key >> 3;
    uint64_t wire_type = key & 0x07;

    if (field_number == 1 && wire_type == 2) { // repeated dim
      utils::StringStream dim_buf;
      stream.read_string_stream(dim_buf);
      Dimension d;
      d.ParseFromString(dim_buf);
      dim.emplace_back(d);
    } else {
      EXT_THROW("[TensorShapeProto::ParseFromString] unknown field number: ", field_number);
    }
  }
}

void TensorProto::ParseFromString(utils::BinaryStream &stream) {
  uint64_t len;
  while (stream.not_end()) {
    uint64_t key = stream.next_uint64();
    uint32_t field_number = key >> 3;
    uint32_t wire_type = key & 0x07;

    switch (field_number) {
    case 1: // dims (repeated int64, varint)
      EXT_ENFORCE(wire_type == 0,
                  "[TensorProto::ParseFromString] dims: wrong wire type (dims)");
      dims.push_back(static_cast<int64_t>(stream.next_uint64()));
      break;

    case 2: // data_type (int32, varint)
      EXT_ENFORCE(wire_type == 0, "[TensorProto::ParseFromString] data_type: wrong wire type");
      data_type = static_cast<TensorProto::DataType>(stream.next_uint64());
      break;

    case 8: // name
      EXT_ENFORCE(wire_type == 2, "[TensorProto::ParseFromString] name: wrong wire type");
      name = stream.next_string();
      break;

    case 9: // raw_data (bytes)
      // Maybe we should avoid a copy here.
      EXT_ENFORCE(wire_type == 2, "[TensorProto::ParseFromString] raw_data: wrong wire type");
      len = stream.next_uint64();
      raw_data.resize(len);
      memcpy(raw_data.data(), stream.read_bytes(len), len);
      break;

    case 12: // doc_string
      EXT_ENFORCE(wire_type == 2, "[TensorProto::ParseFromString] doc_string: wrong wire type");
      doc_string = stream.next_string();
      break;

    case 16: { // metadata_props
      EXT_ENFORCE(wire_type == 2,
                  "[TensorProto::ParseFromString] metadata_props: wrong wire type");
      len = stream.next_uint64();
      stream.can_read(len, "[TensorProto::ParseFromString] metadata_props");
      StringStringEntryProto entry;
      entry.ParseFromString(stream);
      metadata_props.emplace_back(entry);
      break;
    }

    default:
      EXT_THROW("[TensorProto::ParseFromString] unexpected field number: ", field_number);
    }
  }
}

} // namespace onnx2
} // namespace validation
