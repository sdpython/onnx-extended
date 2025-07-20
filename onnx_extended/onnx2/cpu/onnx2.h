#pragma once

#include "onnx_extended_helpers.h"
#include "stream.h"
#include <optional>

#define FIELD_VARINT 0
#define FIELD_FIXED_SIZE 2

/**
 * List of Protos
 * - AttributeProto
 * - DeviceConfigurationProto
 * - GraphProto
 * - FunctionProto
 * - IntIntListEntryProto
 * - ModelProto
 * - NodeDeviceConfigurationProto
 * - NodeProto
 * - OperatorSetIdProto
 * - OperatorStatus
 * - ShardedDimProto
 * - ShardingSpecProto
 * - SimpleShardedDimProto
 * - StringStringEntryProto
 * - TensorAnnotation
 * - TensorProto
 *     - TensorProto::DataLocation
 *     - TensorProto::DataType
 *     - TensorProto::Segment
 * - TensorShapeProto
 *     - TensorShapeProto::Dimension
 * - TrainingInfoProto
 * - TypeProto
 * - ValueInfoProto
 */

#define SERIALIZATION_METHOD()                                                                 \
  inline void ParseFromString(const std::string &raw) {                                        \
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());                        \
    onnx2::utils::StringStream st(ptr, raw.size());                                            \
    ParseFromStream(st);                                                                       \
  }                                                                                            \
  inline void SerializeToString(std::string &out) const {                                      \
    onnx2::utils::StringWriteStream buf;                                                       \
    SerializeToStream(buf);                                                                    \
    out = std::string(reinterpret_cast<const char *>(buf.data()), buf.size());                 \
  }                                                                                            \
  void ParseFromStream(utils::BinaryStream &stream);                                           \
  void SerializeToStream(utils::BinaryWriteStream &stream) const;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif
#define FIELD(type, name, order)                                                               \
public:                                                                                        \
  inline type &name() { return name##_; }                                                      \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  type name##_;

#define FIELD_REPEATED(type, name, order) FIELD(std::vector<type>, name, order)

#define FIELD_OPTIONAL(type, name, order)                                                      \
public:                                                                                        \
  inline type &name() {                                                                        \
    if (name##_.has_value())                                                                   \
      return *name##_;                                                                         \
    EXT_THROW("Optional field '", #name, "' has no value.");                                   \
  }                                                                                            \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  std::optional<type> name##_;

namespace validation {
namespace onnx2 {

using utils::offset_t;

template <typename T> inline bool _has_field_(const T &) { return true; }
template <> inline bool _has_field_<std::string>(const std::string &field) {
  return !field.empty();
}
template <>
inline bool _has_field_<std::optional<uint64_t>>(const std::optional<uint64_t> &field) {
  return field.has_value();
}

class StringStringEntryProto {
public:
  inline StringStringEntryProto() {}
  FIELD(std::string, key, 1)
  FIELD(std::string, value, 2)
  SERIALIZATION_METHOD()
};

class TensorShapeProto {
public:
  class Dimension {
  public:
    inline Dimension() {}
    FIELD_OPTIONAL(uint64_t, dim_value, 1)
    FIELD(std::string, dim_param, 2)
    FIELD(std::string, denotation, 3)
    SERIALIZATION_METHOD()
  };

  inline TensorShapeProto() {}
  FIELD_REPEATED(Dimension, dim, 1)
  SERIALIZATION_METHOD()
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

  enum DataLocation { DEFAULT = 0, EXTERNAL = 1 };

  class Segment {
    FIELD(int64_t, begin, 1)
    FIELD(int64_t, end, 1)
  };

  // methods
  inline TensorProto() { data_type_ = DataType::UNDEFINED; }
  SERIALIZATION_METHOD()

  // data
  FIELD_REPEATED(uint64_t, dims, 1)
  FIELD(DataType, data_type, 2)
  FIELD(Segment, segment, 3)
  FIELD_REPEATED(float, float_data, 4)
  FIELD_REPEATED(int32_t, int32_data, 5)
  FIELD_REPEATED(std::string, string_data, 6)
  FIELD_REPEATED(int64_t, int64_data, 7)
  FIELD(std::string, name, 8)
  FIELD(std::vector<uint8_t>, raw_data, 9)
  FIELD_REPEATED(double, double_data, 10)
  FIELD_REPEATED(uint64_t, uint64_data, 11)
  FIELD(std::string, doc_string, 12)
  FIELD_REPEATED(StringStringEntryProto, external_data, 13)
  FIELD(DataLocation, data_location, 14)
  FIELD_REPEATED(StringStringEntryProto, metadata_props, 16)
};

} // namespace onnx2
} // namespace validation

#if defined(FIELD)
#undef FIELD
#undef FIELD_REPEATED
#endif
