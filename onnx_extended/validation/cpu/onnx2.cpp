#pragma once

#include "onnx2.h"
#include <cstddef>
#include <stdint.h>
#include <vector>
#include <stdexcept>

namespace validation {
namespace onnx2 {

uint64_t readVaruint64(const uint8_t *data, size_t &pos, size_t size) {
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

  throw std::runtime_error("readVaruint64: Unable to read an int64.");
}

int64_t readVarint64(const uint8_t *data, size_t &pos, size_t size) {
  uint64_t value = readVaruint64(data, pos, size);
  return decodeZigZag64(value);
}

std::vector<int64_t> readPackedInt64(const uint8_t *data, size_t &pos, size_t size) {
  std::vector<int64_t> values;

  // read size
  uint64_t length = readVarint64(data, pos, size);
  size_t end = pos + length;
  if (end > size)
    throw std::runtime_error("readPackedInt64: unable to read an array of int64_t");

  // read the array
  while (pos < end) {
    uint64_t raw = readVarint64(data, pos, size);
    int64_t value = static_cast<int64_t>(raw);
    values.push_back(value);
  }

  return values;
}

} // namespace onnx2
} // namespace validation
