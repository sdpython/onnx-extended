#pragma once

#include "onnx_extended_helpers.h"
#include "stream.h"
#include "stream_class.h"

namespace onnx2 {

enum OperatorStatus { EXPERIMENTAL = 0, STABLE = 1 };

class StringStringEntryProto : public Message {
public:
  inline StringStringEntryProto() {}
  FIELD(std::string, key, 1)
  FIELD(std::string, value, 2)
  SERIALIZATION_METHOD()
};

class IntIntListEntryProto : public Message {
public:
  inline IntIntListEntryProto() {}
  SERIALIZATION_METHOD()
  FIELD(int64_t, key, 1)
  FIELD_REPEATED(int64_t, value, 2)
};

class TensorAnnotation : public Message {
public:
  inline TensorAnnotation() {}
  SERIALIZATION_METHOD()
  FIELD(std::string, tensor_name, 1)
  FIELD_REPEATED(StringStringEntryProto, quant_parameter_tensor_names, 2)
};

class DeviceConfigurationProto : public Message {
public:
  inline DeviceConfigurationProto() {}
  SERIALIZATION_METHOD()
  FIELD(std::string, name, 1)
  FIELD(int32_t, num_devices, 2)
  FIELD_REPEATED(std::string, device, 3)
};

class SimpleShardedDimProto : public Message {
public:
  inline SimpleShardedDimProto() {}
  SERIALIZATION_METHOD()
  FIELD_OPTIONAL(int64_t, dim_value, 1)
  FIELD(std::string, dim_param, 2)
  FIELD(int64_t, num_shards, 3)
};

class ShardedDimProto : public Message {
public:
  inline ShardedDimProto() : Message(), axis_(0) {}
  SERIALIZATION_METHOD()
  FIELD(int64_t, axis, 1)
  FIELD_REPEATED(SimpleShardedDimProto, simple_sharding, 2)
};

class ShardingSpecProto : public Message {
public:
  inline ShardingSpecProto() {}
  SERIALIZATION_METHOD()
  FIELD(std::string, tensor_name, 1)
  FIELD_REPEATED(int64_t, device, 2)
  FIELD_REPEATED(IntIntListEntryProto, index_to_device_group_map, 3)
  FIELD_REPEATED(ShardedDimProto, sharded_dim, 4)
};

class NodeDeviceConfigurationProto : public Message {
public:
  inline NodeDeviceConfigurationProto() {}
  SERIALIZATION_METHOD()
  FIELD(std::string, configuration_id, 1)
  FIELD_REPEATED(ShardingSpecProto, sharding_spec, 2)
  FIELD_OPTIONAL(int32_t, pipeline_stage, 3)
};

class OperatorSetIdProto : public Message {
public:
  inline OperatorSetIdProto() {}
  FIELD(std::string, domain, 1)
  FIELD(int64_t, version, 2)
  SERIALIZATION_METHOD()
};

class TensorShapeProto : public Message {
public:
  class Dimension : public Message {
  public:
    inline Dimension() : Message() {}
    FIELD_OPTIONAL(int64_t, dim_value, 1)
    FIELD(std::string, dim_param, 2)
    FIELD(std::string, denotation, 3)
    SERIALIZATION_METHOD()
  };

  inline TensorShapeProto() {}
  FIELD_REPEATED(Dimension, dim, 1)
  SERIALIZATION_METHOD()
};

class TensorProto : public Message {
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
  FIELD_REPEATED_PACKED(float, float_data, 4)
  FIELD_REPEATED_PACKED(int32_t, int32_data, 5)
  FIELD_REPEATED(std::string, string_data, 6)
  FIELD_REPEATED_PACKED(int64_t, int64_data, 7)
  FIELD(std::string, name, 8)
  FIELD(std::vector<uint8_t>, raw_data, 9)
  FIELD_REPEATED_PACKED(double, double_data, 10)
  FIELD_REPEATED_PACKED(uint64_t, uint64_data, 11)
  FIELD(std::string, doc_string, 12)
  FIELD_REPEATED(StringStringEntryProto, external_data, 13)
  FIELD(DataLocation, data_location, 14)
  FIELD_REPEATED(StringStringEntryProto, metadata_props, 16)
};

class SparseTensorProto : public Message {
public:
  inline SparseTensorProto() {}
  SERIALIZATION_METHOD()
  FIELD(TensorProto, values, 1)
  FIELD(TensorProto, indices, 2)
  FIELD_REPEATED(int64_t, dims, 3)
};

class TypeProto : public Message {
public:
  class Tensor : public Message {
  public:
    Tensor() : Message() {}
    SERIALIZATION_METHOD()
    FIELD_OPTIONAL(int32_t, elem_type, 1)
    FIELD(TensorShapeProto, shape, 2)
  };

public:
  inline TypeProto() : Message() {}
  SERIALIZATION_METHOD()
  FIELD_OPTIONAL(Tensor, tensor_type, 1)
};

} // namespace onnx2

#if defined(FIELD)
#undef FIELD
#undef FIELD_REPEATED
#endif
