#include "stream.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace onnx2 {
namespace utils {

std::string FieldNumber::string() const {
  return onnx_extended_helpers::MakeString("[field_number=", field_number,
                                           ", wire_type=", wire_type, "]");
}

RefString BinaryStream::next_string() {
  uint64_t length = next_uint64();
  this->can_read(length, "[StringStream::next_string]");
  return RefString(reinterpret_cast<const char *>(read_bytes(length)),
                   static_cast<size_t>(length));
}

int64_t BinaryStream::next_int64() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int64_t>(value);
}

int32_t BinaryStream::next_int32() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int32_t>(value);
}

float BinaryStream::next_float() {
  return *reinterpret_cast<const float *>(read_bytes(sizeof(float)));
}

double BinaryStream::next_double() {
  return *reinterpret_cast<const double *>(read_bytes(sizeof(double)));
}

FieldNumber BinaryStream::next_field() {
  FieldNumber n;
  n.wire_type = next_uint64();
  n.field_number = n.wire_type >> 3;
  n.wire_type = n.wire_type & 0x07;
  return n;
}

void StringStream::can_read(uint64_t len, const char *msg) {
  EXT_ENFORCE(pos_ + static_cast<int64_t>(len) <= size_, msg, " unable to read ", len,
              " bytes, pos_=", pos_, ", size_=", size_);
}

const uint8_t *StringStream::read_bytes(offset_t n_bytes) {
  const uint8_t *res = data_ + pos_;
  pos_ += n_bytes;
  return res;
}

void StringStream::read_string_stream(StringStream &stream) {
  uint64_t length = next_uint64();
  can_read(length, "[StringStream::read_string_stream]");
  const uint8_t *res = data_ + pos_;
  pos_ += length;
  stream.data_ = res;
  stream.pos_ = 0;
  stream.size_ = length;
}

uint64_t StringStream::next_uint64() {
  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10 && pos_ < size_; ++i) {
    uint8_t byte = data_[pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[StringStream::next_uint64] unable to read an int64 at pos=", pos_,
            ", size=", size_);
}

void BinaryWriteStream::write_variant_uint64(uint64_t value) {
  uint8_t v;
  while (value > 127) {
    v = static_cast<uint8_t>((value & 0x7F) | 0x80);
    write_raw_bytes(&v, 1);
    value >>= 7;
  }
  v = static_cast<uint8_t>(value);
  write_raw_bytes(reinterpret_cast<uint8_t *>(&v), 1);
}

void BinaryWriteStream::write_int64(int64_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_int32(int32_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_float(float value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(float));
}

void BinaryWriteStream::write_double(double value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(double));
}

void BinaryWriteStream::write_field_header(uint32_t field_number, uint8_t wire_type) {
  write_variant_uint64((field_number << 3) | wire_type);
}

void BinaryWriteStream::write_string(const std::string &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void BinaryWriteStream::write_string(const String &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void BinaryWriteStream::write_string(const RefString &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void BinaryWriteStream::write_string_stream(const StringWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

void BinaryWriteStream::write_string_stream(const BorrowedWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

void StringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  buffer_.insert(buffer_.end(), ptr, ptr + n_bytes);
}

int64_t StringWriteStream::size() const { return buffer_.size(); }
const uint8_t *StringWriteStream::data() const { return buffer_.data(); }

void BorrowedWriteStream::write_raw_bytes(const uint8_t *, offset_t) {
  EXT_THROW("This method cannot be called on this class (BorrowedWriteStream).")
}

} // namespace utils
} // namespace onnx2
