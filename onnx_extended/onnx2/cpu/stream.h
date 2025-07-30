#pragma once

#include "onnx_extended_helpers.h"
#include "simple_string.h"
#include <cstddef>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace onnx2 {
namespace utils {

typedef int64_t offset_t;

inline int64_t decodeZigZag64(uint64_t n) { return (n >> 1) ^ -(n & 1); }
inline uint64_t encodeZigZag64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

class StringStream;

struct FieldNumber {
  uint64_t field_number;
  uint64_t wire_type;
  std::string string() const;
};



class BinaryStream {
public:
  explicit inline BinaryStream() {}
  // to overwrite
  virtual uint64_t next_uint64() = 0;
  virtual void can_read(uint64_t len, const char *msg) = 0;
  virtual bool not_end() const = 0;
  virtual offset_t tell() const = 0;
  virtual const uint8_t *read_bytes(offset_t n_bytes) = 0;
  virtual void skip_bytes(offset_t n_bytes) = 0;
  virtual void read_string_stream(StringStream &stream) = 0;
  // defines from the previous ones
  virtual RefString next_string();
  virtual int64_t next_int64();
  virtual int32_t next_int32();
  virtual float next_float();
  virtual double next_double();
  virtual FieldNumber next_field();
  template <typename T> void next_packed_element(T &value) {
    value = *reinterpret_cast<const T *>(read_bytes(sizeof(T)));
  }
};

class StringStream : public BinaryStream {
public:
  explicit inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  explicit inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  virtual void can_read(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  virtual void read_string_stream(StringStream &stream) override;
  virtual bool not_end() const override { return pos_ < size_; }
  virtual offset_t tell() const override { return static_cast<offset_t>(pos_); }

private:
  offset_t pos_;
  offset_t size_;
  const uint8_t *data_;
};

class StringWriteStream;
class BorrowedWriteStream;

class BinaryWriteStream {
public:
  explicit inline BinaryWriteStream() {}
  // to overwrite
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) = 0;
  virtual int64_t size() const = 0;
  virtual const uint8_t *data() const = 0;
  // defined from the previous ones
  virtual void write_variant_uint64(uint64_t value);
  virtual void write_int64(int64_t value);
  virtual void write_int32(int32_t value);
  virtual void write_float(float value);
  virtual void write_double(double value);
  virtual void write_string(const std::string &value);
  virtual void write_string(const String &value);
  virtual void write_string(const RefString &value);
  virtual void write_string_stream(const StringWriteStream &stream);
  virtual void write_string_stream(const BorrowedWriteStream &stream);
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);
  template <typename T> void write_packed_element(const T &value) {
    write_raw_bytes(reinterpret_cast<const uint8_t *>(&value), sizeof(T));
  }
  // size
  virtual uint64_t size_field_header(uint32_t field_number, uint8_t wire_type);
  virtual uint64_t VarintSize(uint64_t value);
  virtual uint64_t size_variant_uint64(uint64_t value);
  virtual uint64_t size_int64(int64_t value);
  virtual uint64_t size_int32(int32_t value);
  virtual uint64_t size_float(float value);
  virtual uint64_t size_double(double value);
  virtual uint64_t size_string(const std::string &value);
  virtual uint64_t size_string(const String &value);
  virtual uint64_t size_string(const RefString &value);
  virtual uint64_t size_string_stream(const StringWriteStream &stream);
  virtual uint64_t size_string_stream(const BorrowedWriteStream &stream);
};

class StringWriteStream : public BinaryWriteStream {
public:
  explicit inline StringWriteStream() : BinaryWriteStream(), buffer_() {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t *data() const override;

private:
  std::vector<uint8_t> buffer_;
};

class BorrowedWriteStream : public BinaryWriteStream {
public:
  explicit inline BorrowedWriteStream(const uint8_t *data, int64_t size)
      : BinaryWriteStream(), data_(data), size_(size) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override { return size_; }
  virtual const uint8_t *data() const override { return data_; }

private:
  const uint8_t *data_;
  int64_t size_;
};

} // namespace utils
} // namespace onnx2
