#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {
namespace onnx2 {

inline int64_t decodeZigZag64(uint64_t n) { return (n >> 1) ^ -(n & 1); }

uint64_t readVaruint64(const uint8_t *data, size_t &pos, size_t size);
int64_t readVarint64(const uint8_t *data, size_t &pos, size_t size);

std::vector<int64_t> readPackedInt64(const uint8_t *data, size_t &pos, size_t size);

} // namespace onnx2
} // namespace validation
