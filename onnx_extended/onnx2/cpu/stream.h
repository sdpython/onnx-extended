#pragma once

#include "onnx_extended_helpers.h"
#include <cstddef>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace validation {
namespace onnx2 {
namespace utils {

typedef int64_t offset_t;

inline int64_t decodeZigZag64(uint64_t n) { return (n >> 1) ^ -(n & 1); }

class StringStream;

struct FieldNumber {
  uint64_t field_number;
  uint64_t wire_type;
  std::string string() const;
};

class BinaryStream {
public:
  inline BinaryStream() {}

  // to overwrite
  virtual uint64_t next_uint64() = 0;
  virtual float next_float32() = 0;
  virtual void can_read(uint64_t len, const char *msg) = 0;
  virtual bool not_end() const = 0;
  virtual offset_t tell() const = 0;
  virtual const uint8_t *read_bytes(offset_t n_bytes) = 0;
  virtual void read_string_stream(StringStream &stream) = 0;

  // defines from the previous ones
  virtual std::string next_string();
  virtual int64_t next_int64();
  virtual FieldNumber next_field();
  virtual void next_packed_element(int64_t &);
  virtual void next_packed_element(uint64_t &);

  template <typename T> inline void next_packed_array(std::vector<T> &values) {
    values.clear();

    // read size
    int64_t length = next_uint64();
    can_read(length, "[BinaryStream::next_packed_array]");

    // read the array
    T raw;
    for (; length > 0; --length) {
      next_packed_ekement(raw);
      values.push_back(raw);
    }
  }
};

class StringStream : public BinaryStream {
public:
  inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  virtual void can_read(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual float next_float32() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes) override;
  virtual void read_string_stream(StringStream &stream) override;
  virtual bool not_end() const override { return pos_ < size_; }
  virtual offset_t tell() const override { return static_cast<offset_t>(pos_); }

private:
  offset_t pos_;
  offset_t size_;
  const uint8_t *data_;
};

class StringWriteStream;

class BinaryWriteStream {
public:
  inline BinaryWriteStream() {}
  virtual void write_variant_uint64(uint64_t value);
  virtual void write_string(const std::string &value);
  virtual void write_string_stream(StringWriteStream &stream);
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);

  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) = 0;
  virtual int64_t size() const = 0;
  virtual const uint8_t *data() const = 0;
};

class StringWriteStream : public BinaryWriteStream {
public:
  inline StringWriteStream() : BinaryWriteStream(), buffer_() {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t *data() const override;

private:
  std::vector<uint8_t> buffer_;
};

} // namespace utils
} // namespace onnx2
} // namespace validation
