#include "stream.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace onnx2 {
namespace utils {

std::string FieldNumber::string() const {
  return onnx_extended_helpers::MakeString("[field_number=", field_number, ", wire_type=", wire_type,
                                           "]");
}

RefString BinaryStream::next_string() {
  uint64_t length = next_uint64();
  this->can_read(length, "[StringStream::next_string]");
  return RefString(reinterpret_cast<const char *>(read_bytes(length)), static_cast<size_t>(length));
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

float BinaryStream::next_float() { return *reinterpret_cast<const float *>(read_bytes(sizeof(float))); }

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

void StringStream::skip_bytes(offset_t n_bytes) { pos_ += n_bytes; }

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
  EXT_THROW("[StringStream::next_uint64] unable to read an uint64 at pos=", pos_, ", size=", size_);
}

std::string StringStream::tell_around() const {
  offset_t begin = pos_;
  offset_t end = pos_ + 10 < static_cast<offset_t>(size()) ? pos_ + 10 : static_cast<offset_t>(size());
  RefString ref(reinterpret_cast<const char *>(data_) + begin, end - begin);
  return ref.as_string();
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

uint64_t BinaryWriteStream::size_variant_uint64(uint64_t value) { return VarintSize(value); }

void BinaryWriteStream::write_int64(int64_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int64(int64_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_int32(int32_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int32(int32_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_float(float value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(float));
}

uint64_t BinaryWriteStream::size_float(float) { return sizeof(float); }

void BinaryWriteStream::write_double(double value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(double));
}

uint64_t BinaryWriteStream::size_double(double) { return sizeof(double); }

void BinaryWriteStream::write_field_header(uint32_t field_number, uint8_t wire_type) {
  write_variant_uint64((field_number << 3) | wire_type);
}

uint64_t BinaryWriteStream::VarintSize(uint64_t value) {
  size_t size = 0;
  do {
    size++;
    value >>= 7;
  } while (value != 0);
  return size;
}

uint64_t BinaryWriteStream::size_field_header(uint32_t field_number, uint8_t wire_type) {
  return VarintSize((field_number << 3) | wire_type);
}

void BinaryWriteStream::write_string(const std::string &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const std::string &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const String &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const String &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const RefString &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const RefString &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string_stream(const StringWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const StringWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::write_string_stream(const BorrowedWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const BorrowedWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void StringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  buffer_.insert(buffer_.end(), ptr, ptr + n_bytes);
}

int64_t StringWriteStream::size() const { return buffer_.size(); }
const uint8_t *StringWriteStream::data() const { return buffer_.data(); }

void BorrowedWriteStream::write_raw_bytes(const uint8_t *, offset_t){
    EXT_THROW("This method cannot be called on this class (BorrowedWriteStream).")}

////////
// file
////////

FileWriteStream::FileWriteStream(const std::string &file_path)
    : BinaryWriteStream(), file_path_(file_path), file_stream_(file_path, std::ios::binary) {}

void FileWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  file_stream_.write(reinterpret_cast<const char *>(data), n_bytes);
}

int64_t FileWriteStream::size() const {
  return static_cast<int64_t>(const_cast<std::ofstream &>(file_stream_).tellp());
}

const uint8_t *FileWriteStream::data() const {
  EXT_THROW("This method cannot be called on this class (FileWriteStream).");
}

FileStream::FileStream(const std::string &file_path)
    : lock_(false), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  if (!file_stream_.is_open()) {
    EXT_THROW("Unable to open file: ", file_path);
  }
  file_stream_.seekg(0, std::ios::end);
  std::streampos end = file_stream_.tellg();
  file_stream_.seekg(0);
  size_ = static_cast<offset_t>(end);
}

bool FileStream::is_open() const { return file_stream_.is_open(); }

void FileStream::can_read(uint64_t len, const char *msg) {
  EXT_ENFORCE(static_cast<int64_t>(tell()) + static_cast<int64_t>(len) <= size_, msg,
              " unable to read ", len, " bytes, pos_=", tell(), ", size_=", size_);
}

uint64_t FileStream::next_uint64() {
  EXT_ENFORCE(!is_locked(), "Please unlock the stream before reading new data.");
  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10; ++i) {
    uint8_t byte = read_bytes(1)[0];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[FileStream::next_uint64] unable to read an int64 at pos=", tell(), ", size=", size_);
}

const uint8_t *FileStream::read_bytes(offset_t n_bytes) {
  EXT_ENFORCE(!is_locked(), "Please unlock the stream before reading new data.");
  if (n_bytes > static_cast<offset_t>(buffer_.size()))
    buffer_.resize(n_bytes);
  file_stream_.read(reinterpret_cast<char *>(buffer_.data()), n_bytes);
  return buffer_.data();
}

void FileStream::skip_bytes(offset_t n_bytes) {
  EXT_ENFORCE(!is_locked(), "Please unlock the stream before reading new data.");
  file_stream_.seekg(n_bytes, std::ios::cur);
}

void FileStream::read_string_stream(StringStream &stream) {
  EXT_ENFORCE(!is_locked(), "Please unlock the stream before reading new data.");
  uint64_t length = next_uint64();
  can_read(length, "[FileStream::read_string_stream]");
  read_bytes(length);
  stream.data_ = buffer_.data();
  stream.pos_ = 0;
  stream.size_ = length;
  set_lock(true);
}

bool FileStream::not_end() const { return static_cast<int64_t>(tell()) < size_; }

offset_t FileStream::tell() const {
  return static_cast<offset_t>(const_cast<std::ifstream &>(file_stream_).tellg());
}

std::string FileStream::tell_around() const {
  RefString ref(reinterpret_cast<const char *>(buffer_.data()),
                buffer_.size() < 10 ? buffer_.size() : 10);
  return ref.as_string();
}

} // namespace utils
} // namespace onnx2
