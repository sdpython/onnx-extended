#pragma once

#include "fields.h"
#include "onnx_extended_helpers.h"
#include "stream.h"
#include "stream_class.h"

#define TensorProto_DataType_UNDEFINED UNDEFINED
#define TensorProto_DataType_FLOAT FLOAT
#define TensorProto_DataType_UINT8 UINT8
#define TensorProto_DataType_INT8 INT8
#define TensorProto_DataType_UINT16 UINT16
#define TensorProto_DataType_INT16 INT16
#define TensorProto_DataType_INT32 INT32
#define TensorProto_DataType_INT64 INT64
#define TensorProto_DataType_STRING STRING
#define TensorProto_DataType_BOOL BOOL
#define TensorProto_DataType_FLOAT16 FLOAT16
#define TensorProto_DataType_DOUBLE DOUBLE
#define TensorProto_DataType_UINT32 UINT32
#define TensorProto_DataType_UINT64 UINT64
#define TensorProto_DataType_COMPLEX64 COMPLEX64
#define TensorProto_DataType_COMPLEX128 COMPLEX128
#define TensorProto_DataType_BFLOAT16 BFLOAT16
#define TensorProto_DataType_FLOAT8E4M3FN FLOAT8E4M3FN
#define TensorProto_DataType_FLOAT8E4M3FNUZ FLOAT8E4M3FNUZ
#define TensorProto_DataType_FLOAT8E5M2 FLOAT8E5M2
#define TensorProto_DataType_FLOAT8E5M2FNUZ FLOAT8E5M2FNUZ
#define TensorProto_DataType_UINT4 UINT4
#define TensorProto_DataType_INT4 INT4
#define TensorProto_DataType_FLOAT4E2M1 FLOAT4E2M1
#define TensorProto_DataType_FLOAT8E8M0 FLOAT8E8M0

namespace onnx2 {

enum OperatorStatus { EXPERIMENTAL = 0, STABLE = 1 };

BEGIN_PROTO(StringStringEntryProto)
FIELD_STR(key, 1)
FIELD_STR(value, 2)
END_PROTO()

BEGIN_PROTO(IntIntListEntryProto)
FIELD_DEFAULT(int64_t, key, 1, 0)
FIELD_REPEATED(int64_t, value, 2)
END_PROTO()

BEGIN_PROTO(TensorAnnotation)
FIELD_STR(tensor_name, 1)
FIELD_REPEATED(StringStringEntryProto, quant_parameter_tensor_names, 2)
END_PROTO()

BEGIN_PROTO(DeviceConfigurationProto)
FIELD_STR(name, 1)
FIELD_DEFAULT(int32_t, num_devices, 2, 0)
FIELD_REPEATED(utils::String, device, 3)
END_PROTO()

BEGIN_PROTO(SimpleShardedDimProto)
FIELD_OPTIONAL(int64_t, dim_value, 1)
FIELD_STR(dim_param, 2)
FIELD_DEFAULT(int64_t, num_shards, 3, 0)
END_PROTO()

BEGIN_PROTO_NOINIT(ShardedDimProto)
inline ShardedDimProto() : axis_(0) {}
FIELD(int64_t, axis, 1)
FIELD_REPEATED(SimpleShardedDimProto, simple_sharding, 2)
END_PROTO()

BEGIN_PROTO(ShardingSpecProto)
FIELD_STR(tensor_name, 1)
FIELD_REPEATED(int64_t, device, 2)
FIELD_REPEATED(IntIntListEntryProto, index_to_device_group_map, 3)
FIELD_REPEATED(ShardedDimProto, sharded_dim, 4)
END_PROTO()

BEGIN_PROTO(NodeDeviceConfigurationProto)
FIELD_STR(configuration_id, 1)
FIELD_REPEATED(ShardingSpecProto, sharding_spec, 2)
FIELD_OPTIONAL(int32_t, pipeline_stage, 3)
END_PROTO()

BEGIN_PROTO(OperatorSetIdProto)
FIELD_STR(domain, 1)
FIELD_DEFAULT(int64_t, version, 2, 0)
END_PROTO()

BEGIN_PROTO_NOINIT(TensorShapeProto)
BEGIN_PROTO(Dimension)
FIELD_OPTIONAL(int64_t, dim_value, 1)
FIELD_STR(dim_param, 2)
FIELD_STR(denotation, 3)
END_PROTO()
inline TensorShapeProto() {}
FIELD_REPEATED(Dimension, dim, 1)
END_PROTO()

// TensorProto

BEGIN_PROTO_NOINIT(TensorProto)
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

BEGIN_PROTO(Segment)
FIELD_DEFAULT(int64_t, begin, 1, 0)
FIELD_DEFAULT(int64_t, end, 1, 0)
END_PROTO()

inline TensorProto() { data_type_ = DataType::UNDEFINED; }

FIELD_REPEATED(uint64_t, dims, 1)
FIELD(DataType, data_type, 2)
FIELD(Segment, segment, 3)
FIELD_REPEATED_PACKED(float, float_data, 4)
FIELD_REPEATED_PACKED(int32_t, int32_data, 5)
FIELD_REPEATED(utils::String, string_data, 6)
FIELD_REPEATED_PACKED(int64_t, int64_data, 7)
FIELD_STR(name, 8)
FIELD(std::vector<uint8_t>, raw_data, 9)
FIELD_REPEATED_PACKED(double, double_data, 10)
FIELD_REPEATED_PACKED(uint64_t, uint64_data, 11)
FIELD_STR(doc_string, 12)
FIELD_REPEATED(StringStringEntryProto, external_data, 13)
FIELD(DataLocation, data_location, 14)
FIELD_REPEATED(StringStringEntryProto, metadata_props, 16)
END_PROTO()

// SparseTensorProto

BEGIN_PROTO(SparseTensorProto)
FIELD(TensorProto, values, 1)
FIELD(TensorProto, indices, 2)
FIELD_REPEATED(int64_t, dims, 3)
END_PROTO()

// TypeProto

BEGIN_PROTO_NOINIT(TypeProto)
BEGIN_PROTO(Tensor)
FIELD_OPTIONAL(int32_t, elem_type, 1)
FIELD_OPTIONAL(TensorShapeProto, shape, 2)
END_PROTO()

BEGIN_PROTO(SparseTensor)
FIELD_OPTIONAL(int32_t, elem_type, 1)
FIELD_OPTIONAL(TensorShapeProto, shape, 2)
END_PROTO()

BEGIN_PROTO(Sequence)
FIELD_OPTIONAL(TypeProto, elem_type, 1)
END_PROTO()

BEGIN_PROTO(Optional)
FIELD_OPTIONAL(TypeProto, elem_type, 1)
END_PROTO()

BEGIN_PROTO(Map)
FIELD(int32_t, key_type, 1)
FIELD_OPTIONAL(TypeProto, value_type, 2)
END_PROTO()

inline TypeProto() {}
FIELD_OPTIONAL_ONEOF(Tensor, tensor_type, 1, type)
FIELD_OPTIONAL_ONEOF(Sequence, sequence_type, 4, type)
FIELD_OPTIONAL_ONEOF(Map, map_type, 5, type)
FIELD_STR(denotation, 6)
FIELD_OPTIONAL_ONEOF(SparseTensor, sparse_tensor_type, 8, type)
FIELD_OPTIONAL_ONEOF(Optional, optional_type, 9, type)
inline bool has_type() const {
  return has_tensor_type() || has_sequence_type() || has_map_type() ||
         has_sparse_tensor_type() || has_optional_type();
}
END_PROTO()

using TensorProto_DataType = TensorProto::DataType;

} // namespace onnx2

#include "fields.hpp"

#if defined(FIELD)
#undef FIELD
#undef FIELD_REPEATED
#endif
