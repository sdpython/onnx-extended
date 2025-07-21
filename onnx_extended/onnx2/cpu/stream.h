#pragma once

#include "onnx_extended_helpers.h"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

namespace onnx2 {
namespace utils {

typedef int64_t offset_t;

template <typename T> class RepeatedField {
public:
  inline RepeatedField() {}
  inline void reserve(size_t n) { values.reserve(n); }
  inline void clear() { values.clear(); }
  inline bool empty() const { return values.empty(); }
  inline size_t size() const { return values.size(); }
  inline T &operator[](size_t index) { return values[index]; }
  inline const T &operator[](size_t index) const { return values[index]; }
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", step);
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", start);
    EXT_ENFORCE(stop == size(), "remove_range not implemented for stop=", stop,
                " and size=", size());
    clear();
  }
  inline void extend(const std::vector<T> &v) {
    values.insert(values.end(), v.begin(), v.end());
  }
  inline void extend(const RepeatedField<T> &v) {
    values.insert(values.end(), v.begin(), v.end());
  }
  inline T &add() {
    values.emplace_back(T());
    return values.back();
  }
  inline std::vector<T>::iterator begin() { return values.begin(); }
  inline std::vector<T>::iterator end() { return values.end(); }
  inline std::vector<T>::const_iterator begin() const { return values.begin(); }
  inline std::vector<T>::const_iterator end() const { return values.end(); }
  std::vector<T> values;
};

template <typename T> class OptionalField {
public:
  inline OptionalField() {}
  inline bool has_value() const { return value.has_value(); }
  inline const T &operator*() const { return *value; }
  inline T &operator*() { return *value; }
  inline bool operator==(const OptionalField<T> &v) const { return value == v; }
  inline bool operator==(const T &v) const { return value == v; }
  inline OptionalField<T> &operator=(const T &other) {
    value = other;
    return *this;
  }
  std::optional<T> value;
};

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
  inline BinaryStream() {}

  // to overwrite
  virtual uint64_t next_uint64() = 0;
  virtual void can_read(uint64_t len, const char *msg) = 0;
  virtual bool not_end() const = 0;
  virtual offset_t tell() const = 0;
  virtual const uint8_t *read_bytes(offset_t n_bytes) = 0;
  virtual void read_string_stream(StringStream &stream) = 0;

  // defines from the previous ones
  virtual std::string next_string();
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
  inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  virtual void can_read(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
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
class BorrowedWriteStream;

class BinaryWriteStream {
public:
  inline BinaryWriteStream() {}
  virtual void write_variant_uint64(uint64_t value);
  virtual void write_int64(int64_t value);
  virtual void write_float(float value);
  virtual void write_double(double value);
  virtual void write_string(const std::string &value);
  virtual void write_string_stream(const StringWriteStream &stream);
  virtual void write_string_stream(const BorrowedWriteStream &stream);
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);
  template <typename T> void write_packed_element(const T &value) {
    write_raw_bytes(reinterpret_cast<const uint8_t *>(&value), sizeof(T));
  }

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

class BorrowedWriteStream : public BinaryWriteStream {
public:
  inline BorrowedWriteStream(const uint8_t *data, int64_t size)
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
