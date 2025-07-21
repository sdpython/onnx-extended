#include "onnx2.h"
#include "stream_class.hpp"

namespace onnx2 {

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_FIELD(stream, value)
}

void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, StringStringEntryProto)
  READ_FIELD(stream, key)
  READ_FIELD(stream, value)
  READ_END(stream, StringStringEntryProto)
}

void OperatorSetIdProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, domain)
  WRITE_FIELD(stream, version)
}

void OperatorSetIdProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, OperatorSetIdProto)
  READ_FIELD(stream, domain)
  READ_FIELD(stream, version)
  READ_END(stream, OperatorSetIdProto)
}

void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, dim_value)
  WRITE_FIELD(stream, dim_param)
  WRITE_FIELD(stream, denotation)
}

void TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TensorShapeProto::Dimension)
  READ_FIELD(stream, dim_value)
  READ_FIELD(stream, dim_param)
  READ_FIELD(stream, denotation)
  READ_END(stream, TensorShapeProto::Dimension)
}

void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dim)
}

void TensorShapeProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TensorShapeProto)
  READ_REPEATED_FIELD(stream, dim)
  READ_END(stream, TensorShapeProto)
}

void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dims)
  WRITE_ENUM_FIELD(stream, data_type)
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, raw_data)
  WRITE_FIELD(stream, doc_string)
  WRITE_REPEATED_FIELD(stream, external_data)
  WRITE_REPEATED_FIELD(stream, metadata_props)
  //
  WRITE_REPEATED_FIELD(stream, double_data)
  WRITE_REPEATED_FIELD(stream, float_data)
  WRITE_REPEATED_FIELD(stream, int32_data)
  WRITE_REPEATED_FIELD(stream, int64_data)
  WRITE_REPEATED_FIELD(stream, uint64_data)
  WRITE_REPEATED_FIELD(stream, string_data)
}

void TensorProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TensorProto)
  READ_REPEATED_FIELD(stream, dims)
  READ_ENUM_FIELD(stream, data_type)
  READ_FIELD(stream, name)
  READ_FIELD(stream, doc_string)
  READ_FIELD(stream, raw_data)
  READ_REPEATED_FIELD(stream, external_data)
  READ_REPEATED_FIELD(stream, metadata_props)
  //
  READ_REPEATED_FIELD(stream, double_data)
  READ_REPEATED_FIELD(stream, float_data)
  READ_REPEATED_FIELD(stream, int32_data)
  READ_REPEATED_FIELD(stream, int64_data)
  READ_REPEATED_FIELD(stream, uint64_data)
  READ_REPEATED_FIELD(stream, string_data)
  READ_END(stream, TensorProto)
}

void SparseTensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, values)
  WRITE_FIELD(stream, indices)
  WRITE_REPEATED_FIELD(stream, dims)
}

void SparseTensorProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, SparseTensorProto)
  READ_FIELD(stream, values)
  READ_FIELD(stream, indices)
  READ_REPEATED_FIELD(stream, dims)
  READ_END(stream, SparseTensorProto)
}

} // namespace onnx2
