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

class BinaryStream {
public:
  inline BinaryStream() {}
  virtual uint64_t next_uint64() = 0;
  virtual int64_t next_int64() = 0;
  virtual float next_float32() = 0;
  virtual std::string next_string() = 0;
  virtual void can_read(uint64_t len, const char *msg) = 0;
  virtual bool not_end() const = 0;
  virtual offset_t tell() const = 0;
  virtual const uint8_t *read_bytes(offset_t n_bytes) = 0;
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
  inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  virtual void can_read(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual int64_t next_int64() override;
  virtual float next_float32() override;
  virtual std::string next_string() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes) override;
  virtual bool not_end() const override { return pos_ < size_; }
  virtual offset_t tell() const override { return static_cast<offset_t>(pos_); }

private:
  offset_t pos_;
  offset_t size_;
  const uint8_t *data_;
};

} // namespace utils

using utils::offset_t;

class StringStringEntryProto {
public:
  std::string key;
  std::string value;
  inline StringStringEntryProto() {}
  void ParseFromString(utils::BinaryStream &stream);
};

class TensorProto {
public:
  enum class DataType : int32_t {
    UNDEFINED = 0,
    // Basic types.
    FLOAT = 1,  // float
    UINT8 = 2,  // uint8_t
    INT8 = 3,   // int8_t
    UINT16 = 4, // uint16_t
    INT16 = 5,  // int16_t
    INT32 = 6,  // int32_t
    INT64 = 7,  // int64_t
    STRING = 8, // string
    BOOL = 9,   // bool

    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10,

    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,  // complex with float32 real and imaginary components
    COMPLEX128 = 15, // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16,

    // Non-IEEE floating-point format based on papers
    // FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
    // 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
    // Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
    // The computation usually happens inside a block quantize / dequantize
    // fused by the runtime.
    FLOAT8E4M3FN = 17,   // float 8, mostly used for coefficients, nan, no inf
    FLOAT8E4M3FNUZ = 18, // float 8, mostly used for coefficients, nan, no inf, no negative zero
    FLOAT8E5M2 = 19,     // follows IEEE 754, supports nan, inf
    FLOAT8E5M2FNUZ = 20, // follows IEEE 754, supports nan, no inf, no negative zero
                         // no negative zero

    // 4-bit integer data types
    UINT4 = 21, // Unsigned integer in range [0, 15]
    INT4 = 22,  // Signed integer in range [-8, 7], using two's-complement representation

    // 4-bit floating point data types
    FLOAT4E2M1 = 23,

    // E8M0 type used as the scale for microscaling (MX) formats:
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    FLOAT8E8M0 = 24

    // Future extensions go here.
  };

  struct Segment {
    int64_t begin;
    int64_t end;
  };

  // methods
  inline TensorProto() { data_type = DataType::UNDEFINED; }
  void ParseFromString(utils::BinaryStream &stream);

  // data
  std::vector<int64_t> dims;                         // 1
  DataType data_type;                                // 2
  Segment segment;                                   // 3
  std::vector<float> float_data;                     // 4, packed
  std::vector<int32_t> int32_data;                   // 5, packed
  std::vector<uint8_t> string_data;                  // 6
  std::vector<int64_t> int64_data;                   // 7, packed
  std::string name;                                  // 8
  std::vector<uint8_t> raw_data;                     // 9
  std::vector<double> double_data;                   // 10, packed
  std::vector<uint64_t> uint64_data;                 // 11, packed
  std::string doc_string;                            // 12
  std::vector<StringStringEntryProto> external_data; // 13

  enum DataLocation { DEFAULT = 0, EXTERNAL = 1 };

  DataLocation data_location; // 14

  std::vector<StringStringEntryProto> metadata_props; // 16
};

} // namespace onnx2
} // namespace validation
