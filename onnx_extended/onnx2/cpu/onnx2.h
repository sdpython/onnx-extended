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

BEGIN_PROTO(StringStringEntryProto, "Defines a key value pair, both defines a strings.")
FIELD_STR(key, 1, "the key")
FIELD_STR(value, 2, "the value")
END_PROTO()

BEGIN_PROTO(IntIntListEntryProto,
            "Defines a key value pair, key is an integer, value is a list of integers.")
FIELD_DEFAULT(int64_t, key, 1, 0, "the key")
FIELD_REPEATED(int64_t, value, 2, "the value is a list of integers")
END_PROTO()

BEGIN_PROTO(TensorAnnotation, "Defines a tensor annotation, useful for quantized tensors.")
FIELD_STR(tensor_name, 1, "tensor name")
FIELD_REPEATED(StringStringEntryProto, quant_parameter_tensor_names, 2,
               "<key, value> pairs to annotate tensor specified by <tensor_name> above. The "
               "keys used in the mapping below must be pre-defined in ONNX spec. For example, "
               "for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will "
               "be pre-defined as quantization parameter keys.")
END_PROTO()

BEGIN_PROTO(DeviceConfigurationProto, "Describes a multi-device configuration for a model.")
FIELD_STR(name, 1,
          "This field MUST be present for this version of the IR. Name of the configuration.")
FIELD_DEFAULT(int32_t, num_devices, 2, 0,
              "This field MUST be present for this version of the IR. Number of devices inside "
              "this configuration.")
FIELD_REPEATED(utils::String, device, 3,
               "Optional names of the devices. MUST be length of num_devices if provided.")
END_PROTO()

BEGIN_PROTO(SimpleShardedDimProto,
            "Indicates that N blocks are divided into M shards. N is allowed to be symbolic "
            "where M is required to be a constant.")
FIELD_OPTIONAL(int64_t, dim_value, 1, "Dimension value to be sharded if it is a fixed value.")
FIELD_STR(dim_param, 2, "Dimension value to be sharded if it is a dynamic value.")
FIELD_DEFAULT(int64_t, num_shards, 3, 0,
              "This field MUST be present for this version of the IR. Number of shards to "
              "split dim into.")
END_PROTO()

BEGIN_PROTO_NOINIT(ShardedDimProto,
                   "Describes the sharding spec for a single axis of a sharded tensor.")
inline ShardedDimProto() : axis_(0) {}
FIELD(int64_t, axis, 1,
      "This field MUST be present for this version of the IR. The axis this sharding "
      "corresponds to. Must be in the range of [-r, r - 1], where r is the rank of the tensor. "
      "Negative axis values means counting from the back.")
FIELD_REPEATED(
    SimpleShardedDimProto, simple_sharding, 2,
    "Describes how the tensor on the provided axis is sharded. The common-case is described by "
    "a single instance of SimpleShardedDimProto. Multiple instances can be used to handle "
    "cases where a sharded tensor is reshaped, fusing multiple axes into one.")
END_PROTO()

BEGIN_PROTO(ShardingSpecProto,
            "Describes the sharding spec for a specific, input or output tensor of a node.")
FIELD_STR(
    tensor_name, 1,
    "This field MUST be present for this version of the IR. Identifies the input or output of "
    "the node that is being sharded. Required to match a name specified in the node's input or "
    "output list of ValueInfoProtos. It is called `logical tensor` in subsequent descriptions.")
FIELD_REPEATED(int64_t, device, 2,
               "The following is the list of devices across which the logical tensor is "
               "sharded or replicated.")
FIELD_REPEATED(
    IntIntListEntryProto, index_to_device_group_map, 3,
    "Each element v in above field devices may represent either a device or a set of devices "
    "(when we want the same shard/tensor to be replicated across a subset of devices), as "
    "indicated by the following optional map. If the map contains an entry for v, then v "
    "represents a device group, and the map indicates the set of devices in that group.")
FIELD_REPEATED(ShardedDimProto, sharded_dim, 4,
               "The following is the sharded-shape of the tensor, consisting of the "
               "sharding-spec for each axis of the tensor.")
END_PROTO()

BEGIN_PROTO(NodeDeviceConfigurationProto,
            "Defines a multi-device configuration proto for NodeProto.")
FIELD_STR(configuration_id, 1,
          "This field MUST be present for this version of the IR., ID of the configuration. "
          "MUST match the name of a DeviceConfigurationProto.")
FIELD_REPEATED(ShardingSpecProto, sharding_spec, 2, "Sharding spec for the node.")
FIELD_OPTIONAL(int32_t, pipeline_stage, 3, "Pipeline stage of this node.")
END_PROTO()

BEGIN_PROTO(OperatorSetIdProto,
            "Defines a unqiue pair domain, opset version for a set of operators.")
FIELD_STR(domain, 1,
          "The domain of the operator set being identified. The empty string ("
          ") or absence of this field implies the operator set that is defined as part of the "
          "ONNX specification. This field MUST be present in this version of the IR when "
          "referring to any other operator set.")
FIELD_DEFAULT(int64_t, version, 2, 0,
              "The version of the operator set being identified. This field MUST be present in "
              "this version of the IR.")
END_PROTO()

BEGIN_PROTO_NOINIT(TensorShapeProto,
                   "Defines a tensor shape. A dimension can be either an integer value or a "
                   "symbolic variable. A symbolic variable represents an unknown dimension.")
BEGIN_PROTO(Dimension, "Defines a dimension, it can be fixed (an integer dim_value) or dynamic "
                       "(a string dim_param). Only one of them can be set.")
FIELD_OPTIONAL(int64_t, dim_value, 1, "Dimension value if it is a fixed value.")
FIELD_STR(dim_param, 2, "Dimension value if it is a dynamic value.")
FIELD_STR(
    denotation, 3,
    "The indices of the non-default values, which may be stored in one of two formats. (a) "
    "Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value corresponding to the "
    "j-th index of the i-th value (in the values tensor). (b) Indices can be a tensor of shape "
    "[NNZ], in which case the i-th value must be the linearized-index of the i-th value (in "
    "the values tensor). The linearized-index can be converted into an index tuple "
    "(k_1,...,k_rank) using the shape provided below. The indices must appear in ascending "
    "order without duplication. In the first format, the ordering is lexicographic-ordering: "
    "e.g., index-value [1,4] must appear before [2,1]")
END_PROTO()
inline TensorShapeProto() {}
FIELD_REPEATED(Dimension, dim, 1, "Shape as a list of Dimension.")
END_PROTO()

// TensorProto

BEGIN_PROTO_NOINIT(TensorProto, "Defines a tensor and its content.")
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

BEGIN_PROTO(
    Segment,
    "For very large tensors, we may want to store them in chunks, in which case the following "
    "fields will specify the segment that is stored in the current TensorProto.")
FIELD_DEFAULT(int64_t, begin, 1, 0, "Segment start.")
FIELD_DEFAULT(int64_t, end, 1, 0, "Segment end.")
END_PROTO()

inline TensorProto() { data_type_ = DataType::UNDEFINED; }

FIELD_REPEATED(uint64_t, dims, 1, "The shape of the tensor.")
FIELD(DataType, data_type, 2,
      "The data type of the tensor. This field MUST have a valid TensorProto.DataType value")
FIELD(Segment, segment, 3,
      "For very large tensors, we may want to store them in chunks, in which case the "
      "following fields will specify the segment that is stored in the current TensorProto.")
FIELD_REPEATED_PACKED(
    float, float_data, 4,
    "Tensor content must be organized in row-major order. Depending on the data_type field, "
    "exactly one of the fields below with name ending in _data is used to store the elements "
    "of the tensor. For float and complex64 values Complex64 tensors are encoded as a single "
    "array of floats, with the real components appearing in odd numbered positions, and the "
    "corresponding imaginary component appearing in the subsequent even numbered position. "
    "(e.g., [1.0 + 2.0i, 3.0 + 4.0i] is encoded as [1.0, 2.0 ,3.0 ,4.0] When this field is "
    "present, the data_type field MUST be FLOAT or COMPLEX64.")
FIELD_REPEATED_PACKED(
    int32_t, int32_data, 5,
    "For int32, uint8, int8, uint16, int16, uint4, int4, bool, (b)float16, float8, and "
    "float4: - (b)float16 and float8 values MUST be converted bit-wise into an unsigned "
    "integer representation before being written to the buffer. - Each pair of uint4, int4, "
    "and float4 values MUST be packed as two 4-bit elements into a single byte. The first "
    "element is stored in the 4 least significant bits (LSB), and the second element is "
    "stored in the 4 most significant bits (MSB). Consequently: - For data types with a "
    "bit-width of 8 or greater, each `int32_data` stores one element. - For 4-bit data "
    "types, each `int32_data` stores two elements. When this field is present, the data_type "
    "field MUST be INT32, INT16, INT8, INT4, UINT16, UINT8, UINT4, BOOL, FLOAT16, BFLOAT16, "
    "FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, FLOAT8E8M0, FLOAT4E2M1")
FIELD_REPEATED(utils::String, string_data, 6,
               "For strings. Each element of string_data is a UTF-8 encoded Unicode string. No "
               "trailing null, no leading BOM. The 'string' scalar type is not used to match "
               "ML community conventions. When this field is "
               "present, the data_type field MUST be STRING")
FIELD_REPEATED_PACKED(
    int64_t, int64_data, 7,
    "For int64. When this field is present, the data_type field MUST be INT64")
FIELD_STR(name, 8, "Optionally, a name for the tensor.")
FIELD(std::vector<uint8_t>, raw_data, 9,
      "Serializations can either use one of the fields above, or use this raw bytes field. The "
      "only exception is the string case, where one is required to store the content in the "
      "repeated bytes string_data field. When this raw_data field is used to store tensor "
      "value, elements MUST be stored in as fixed-width, little-endian order. Floating-point "
      "data types MUST be stored in IEEE 754 format. Complex64 elements must be written as two "
      "consecutive FLOAT values, real component first. Complex128 elements must be written as "
      "two consecutive DOUBLE values, real component first. Boolean type MUST be written one "
      "byte per tensor element (00000001 for true, 00000000 for false). uint4 and int4 values "
      "must be packed to 4bitx2, the first element is stored in the 4 LSB and the second "
      "element is stored in the 4 MSB. Note: the advantage of specific field rather than the "
      "raw_data field is that in some cases (e.g. int data), protobuf does a better packing "
      "via variable length storage, and may lead to smaller binary footprint. When this field "
      "is present, the data_type field MUST NOT be STRING or UNDEFINED.")
FIELD_REPEATED_PACKED(
    double, double_data, 10,
    "For double Complex128 tensors are encoded as a single array of doubles, with the real "
    "components appearing in odd numbered positions, and the corresponding imaginary component "
    "appearing in the subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i] is "
    "encoded as [1.0, 2.0 ,3.0 ,4.0] When this field is present, the data_type field MUST be "
    "DOUBLE or COMPLEX128.")
FIELD_REPEATED_PACKED(uint64_t, uint64_data, 11,
                      "For uint64 and uint32 values. When this field is present, the data_type "
                      "field MUST be UINT32 or UINT64.")
FIELD_STR(doc_string, 12,
          "A human-readable documentation for this tensor. Markdown is allowed.")
FIELD_REPEATED(
    StringStringEntryProto, external_data, 13,
    "Data can be stored inside the protobuf file using type-specific fields or raw_data. "
    "Alternatively, raw bytes data can be stored in an external file, using the external_data "
    "field. external_data stores key-value pairs describing data location. Recognized keys "
    "are: "
    "- location (required) - POSIX filesystem path relative to the directory where "
    "the ONNX protobuf model was stored "
    "- offset (optional) - position of byte at which stored data begins. Integer stored as "
    "string. "
    "Offset values SHOULD be multiples 4096 (page size) to enable mmap support. "
    "- length (optional) - number of bytes containing data. Integer stored as string. "
    "- checksum (optional) - SHA1 digest of file specified in under 'location' key.")
FIELD(
    DataLocation, data_location, 14,
    "Location of the data for this tensor. MUST be one of: - DEFAULT - data stored inside the "
    "protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field. "
    "- EXTERNAL - data stored in an external location as described by external_data field. If "
    "value not set, data is stored in raw_data (if set) otherwise in type-specified field.")
FIELD_REPEATED(StringStringEntryProto, metadata_props, 16,
               "Named metadata values; keys should be distinct.")
END_PROTO()

// SparseTensorProto

BEGIN_PROTO(SparseTensorProto, "A sparse tensor.")
FIELD(TensorProto, values, 1,
      "The sequence of non-default values are encoded as a tensor of shape [NNZ]. The "
      "default-value is zero for numeric tensors, and empty-TypeProto::Tensortring for string "
      "tensors. values must have a non-empty name present which serves as a name for "
      "SparseTensorProto when used in sparse_initializer list.")
FIELD(TensorProto, indices, 2,
      "The indices of the non-default values, which may be stored in one of two formats. (a) "
      "Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value corresponding to "
      "the j-th index of the i-th value (in the values tensor). (b) Indices can be a tensor of "
      "shape [NNZ], in which case the i-th value must be the linearized-index of the i-th "
      "value (in the values tensor). The linearized-index can be converted into an index tuple "
      "(k_1,...,k_rank) using the shape provided below. The indices must appear in ascending "
      "order without duplication. In the first format, the ordering is lexicographic-ordering: "
      "e.g., index-value [1,4] must appear before [2,1].")
FIELD_REPEATED(int64_t, dims, 3, "The shape.")
END_PROTO()

// TypeProto

BEGIN_PROTO_NOINIT(TypeProto, "Defines a type, it can be a tensor type (element type and "
                              "shape), a sequence of the same element type, ...")
BEGIN_PROTO(Tensor, "Defines a tensor type (element type, shape).")
FIELD_OPTIONAL(
    int32_t, elem_type, 1,
    "This field MUST NOT have the value of UNDEFINED. This field MUST have a valid "
    "TensorProto.DataType value. This field MUST be present for this version of the IR.")
FIELD_OPTIONAL(TensorShapeProto, shape, 2, "The shape.")
END_PROTO()

BEGIN_PROTO(SparseTensor, "Defines a sparse tensor type (element type, shape)")
FIELD_OPTIONAL(
    int32_t, elem_type, 1,
    "This field MUST NOT have the value of UNDEFINThis field MUST NOT have the value of "
    "UNDEFINED. This field MUST have a valid TensorProto.DataType value. This field MUST be "
    "present for this version of the IR.ED. This field MUST have a valid TensorProto.DataType "
    "value. This field MUST be present for this version of the IR.")
FIELD_OPTIONAL(TensorShapeProto, shape, 2, "The shape.")
END_PROTO()

BEGIN_PROTO(Sequence, "Defines the type of each element in a sequence.")
FIELD_OPTIONAL(TypeProto, elem_type, 1,
               "The type and optional shape of each element of the sequence. This field MUST "
               "be present for this version of the IR.")
END_PROTO()

BEGIN_PROTO(Optional, "Defines the type of an optional value.")
FIELD_OPTIONAL(
    TypeProto, elem_type, 1,
    "The type and optional shape of the element wrapped. This field MUST be present for this "
    "version of the IR. Possible values correspond to OptionalProto.DataType enum")
END_PROTO()

BEGIN_PROTO(Map, "Defines the type of the key and the type of each value in a dictionary.")
FIELD(int32_t, key_type, 1,
      "This field MUST have a valid TensorProto.DataType value. This field MUST be present for "
      "this version of the IR. This field MUST refer to an integral type ([U]INT{8|16|32|64}) "
      "or STRING optional int32 key_type = 1;")
FIELD_OPTIONAL(TypeProto, value_type, 2,
               "This field MUST be present for this version of the IR.")
END_PROTO()

inline TypeProto() {}
FIELD_OPTIONAL_ONEOF(Tensor, tensor_type, 1, type, "The type of a tensor.")
FIELD_OPTIONAL_ONEOF(Sequence, sequence_type, 4, type, "The type of a sequence.")
FIELD_OPTIONAL_ONEOF(Map, map_type, 5, type, "The type of a map.")
FIELD_STR(denotation, 6,
          "An optional denotation can be used to denote the whole type with a standard "
          "semantic description as to what is stored inside. Refer to "
          "https://github.com/onnx/onnx/blob/main/docs/"
          "TypeDenotation.md#type-denotation-definition for pre-defined type denotations.")
FIELD_OPTIONAL_ONEOF(SparseTensor, sparse_tensor_type, 8, type, "Type of the sparse tensor")
FIELD_OPTIONAL_ONEOF(Optional, optional_type, 9, type, "The type of an optional.")
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
