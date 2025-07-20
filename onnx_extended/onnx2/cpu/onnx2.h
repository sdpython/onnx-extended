#pragma once

#include "onnx_extended_helpers.h"
#include "stream.h"

#define FIELD_VARINT 0
#define FIELD_FIXED_SIZE 2

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

namespace validation {
namespace onnx2 {

using utils::offset_t;

class StringStringEntryProto {
public:
  std::string key;
  std::string value;
  inline StringStringEntryProto() {}
  SERIALIZATION_METHOD()
};

class TensorShapeProto {
public:
  class Dimension {
  public:
    inline Dimension() : dim_value(0), dim_param(), denotation() {}
    int64_t dim_value;      // 1
    std::string dim_param;  // 2
    std::string denotation; // 3
    SERIALIZATION_METHOD()
  };

  inline TensorShapeProto() : dim() {}
  SERIALIZATION_METHOD()

  std::vector<Dimension> dim; // 1
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
