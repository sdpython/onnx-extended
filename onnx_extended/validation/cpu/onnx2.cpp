#include "onnx2.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace validation {
namespace onnx2 {
namespace utils {

uint64_t readVaruint64(const uint8_t *data, offset_t &pos, offset_t size) {
  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10 && pos < size; ++i) {
    uint8_t byte = data[pos++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0) {
      return result;
    }

    shift += 7;
  }
  EXT_THROW("readVaruint64: unable to read an int64 at pos=", pos, ", size=", size);
}

int64_t readVarint64(const uint8_t *data, offset_t &pos, offset_t size) {
  uint64_t value = readVaruint64(data, pos, size);
  return decodeZigZag64(value);
}

float readFloat32(const uint8_t *data, offset_t &pos, offset_t size) {
  EXT_ENFORCE(size >= sizeof(float) + pos, "readFloat32, not enough bytes to read, pos=", pos,
              ", size=", size);
  float value;
  memcpy(&value, data + pos, sizeof(float));
  pos += sizeof(float);
  return value;
}

std::string readStringField(const uint8_t *data, offset_t &pos, offset_t size) {
  uint64_t length = readVaruint64(data, pos, size);
  EXT_ENFORCE(pos + length <= size, "readStringField: buffer too short, pos=", pos,
              ", length=", length, ", size=", size);
  std::string result(reinterpret_cast<const char *>(data + pos), length);
  pos += length;
  return result;
}

} // namespace utils

void StringStringEntryProto::ParseFromString(const uint8_t *data, offset_t &pos,
                                             offset_t size) {
  key.clear();
  value.clear();
  while (pos < size) {
    uint64_t field_key = utils::readVaruint64(data, pos, size);
    uint32_t field_number = field_key >> 3;
    uint32_t wire_type = field_key & 0x07;
    EXT_ENFORCE(wire_type == 2,
                "StringStringEntryProto::ParseFromString: expected length-delimited wire type");
    if (field_number == 1) {
      key = utils::readStringField(data, pos, size);
    } else if (field_number == 2) {
      value = utils::readStringField(data, pos, size);
    } else {
      EXT_THROW("StringStringEntryProto::ParseFromString: unknown field number: ",
                field_number);
    }
  }
}

void TensorProto::ParseFromString(const uint8_t *data, offset_t &pos, offset_t size) {
  uint64_t len;
  while (pos < size) {
    uint64_t key = utils::readVaruint64(data, pos, size);
    uint32_t field_number = key >> 3;
    uint32_t wire_type = key & 0x07;

    switch (field_number) {
    case 1: // dims (repeated int64, varint)
      EXT_ENFORCE(wire_type == 0, "TensorProto::ParseFromString: dims: wrong wire type (dims)");
      dims.push_back(static_cast<int64_t>(utils::readVaruint64(data, pos, size)));
      break;

    case 2: // data_type (int32, varint)
      EXT_ENFORCE(wire_type == 0, "TensorProto::ParseFromString: data_type: wrong wire type");
      data_type = static_cast<TensorProto::DataType>(utils::readVaruint64(data, pos, size));
      break;

    case 8: // name
      EXT_ENFORCE(wire_type == 2, "TensorProto::ParseFromString: name: wrong wire type");
      len = utils::readVaruint64(data, pos, size);
      EXT_ENFORCE(pos + len <= size,
                  "TensorProto::ParseFromString: name: length out of bounds, pos=", pos,
                  ", len=", len, ", size=", size);
      name = std::string(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      break;

    case 9: // raw_data (bytes)
      EXT_ENFORCE(wire_type == 2, "TensorProto::ParseFromString: raw_data: wrong wire type");
      len = utils::readVaruint64(data, pos, size);
      raw_data.resize(len);
      memcpy(raw_data.data(), data + pos, len);
      pos += len;
      break;

    case 12: // doc_string
      EXT_ENFORCE(wire_type == 2, "TensorProto::ParseFromString: doc_string: wrong wire type");
      len = utils::readVaruint64(data, pos, size);
      EXT_ENFORCE(pos + len <= size,
                  "TensorProto::ParseFromString: doc_string: length out of bounds, pos=", pos,
                  ", len=", len, ", size=", size);
      doc_string = std::string(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
      break;

    case 16: { // metadata_props
      EXT_ENFORCE(wire_type == 2,
                  "TensorProto::ParseFromString: metadata_props: wrong wire type");
      len = utils::readVaruint64(data, pos, size);
      EXT_ENFORCE(pos + len <= size,
                  "TensorProto::ParseFromString: metadata_props: length out of bounds, pos=",
                  pos, ", len=", len, ", size=", size);
      StringStringEntryProto entry;
      entry.ParseFromString(data, pos, pos + len);
      metadata_props.emplace_back(entry);
      break;
    }

    default:
      EXT_THROW("TensorProto::ParseFromString: unexpected field number: ", field_number);
    }
  }
}

} // namespace onnx2
} // namespace validation
