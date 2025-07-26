#include "onnx2.h"
#include "stream_class.hpp"
#include <sstream>

namespace onnx2 {

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_FIELD(stream, value)
}

void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, StringStringEntryProto) READ_FIELD(stream, key) READ_FIELD(stream, value)
        READ_END(stream, StringStringEntryProto)}

std::vector<std::string> StringStringEntryProto::SerializeToVectorString() const {
  return {
      write_as_string(std::make_pair(_name_key, key()), std::make_pair(_name_value, value()))};
}

void TensorAnnotation::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, tensor_name)
  WRITE_REPEATED_FIELD(stream, quant_parameter_tensor_names)
}

void TensorAnnotation::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorAnnotation) READ_FIELD(stream, tensor_name)
        READ_REPEATED_FIELD(stream, quant_parameter_tensor_names)
            READ_END(stream, TensorAnnotation)}

std::vector<std::string> TensorAnnotation::SerializeToVectorString() const {
  return write_proto_into_vector_string(
      std::make_pair(_name_tensor_name, tensor_name()),
      std::make_pair(_name_quant_parameter_tensor_names, quant_parameter_tensor_names()));
}

void IntIntListEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_REPEATED_FIELD(stream, value)
}

void IntIntListEntryProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, IntIntListEntryProto) READ_FIELD(stream, key)
        READ_REPEATED_FIELD(stream, value) READ_END(stream, IntIntListEntryProto)}

std::vector<std::string> IntIntListEntryProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(std::make_pair(_name_key, key()),
                                        std::make_pair(_name_value, value()));
}

void DeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, num_devices)
  WRITE_REPEATED_FIELD(stream, device)
}

void DeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, DeviceConfigurationProto) READ_FIELD(stream, name)
        READ_FIELD(stream, num_devices) READ_REPEATED_FIELD(stream, device)
            READ_END(stream, DeviceConfigurationProto)}

std::vector<std::string> DeviceConfigurationProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(std::make_pair(_name_name, name()),
                                        std::make_pair(_name_num_devices, num_devices()),
                                        std::make_pair(_name_device, device()));
}

void SimpleShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, dim_value)
  WRITE_FIELD(stream, dim_param)
  WRITE_FIELD(stream, num_shards)
}

void SimpleShardedDimProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, SimpleShardedDimProto)
  READ_FIELD(stream, dim_value)
  READ_FIELD(stream, dim_param)
  READ_FIELD(stream, num_shards)
  READ_END(stream, SimpleShardedDimProto)
}

void ShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, axis)
  WRITE_REPEATED_FIELD(stream, simple_sharding)
}

void ShardedDimProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, ShardedDimProto)
  READ_FIELD(stream, axis)
  READ_REPEATED_FIELD(stream, simple_sharding)
  READ_END(stream, ShardedDimProto)
}

void ShardingSpecProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, tensor_name)
  WRITE_REPEATED_FIELD(stream, device)
  WRITE_REPEATED_FIELD(stream, index_to_device_group_map)
  WRITE_REPEATED_FIELD(stream, sharded_dim)
}

void ShardingSpecProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, ShardingSpecProto)
  READ_FIELD(stream, tensor_name)
  READ_REPEATED_FIELD(stream, device)
  READ_REPEATED_FIELD(stream, index_to_device_group_map)
  READ_REPEATED_FIELD(stream, sharded_dim)
  READ_END(stream, ShardingSpecProto)
}

void NodeDeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, configuration_id)
  WRITE_REPEATED_FIELD(stream, sharding_spec)
  WRITE_FIELD(stream, pipeline_stage)
}

void NodeDeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, NodeDeviceConfigurationProto)
  READ_FIELD(stream, configuration_id)
  READ_REPEATED_FIELD(stream, sharding_spec)
  READ_FIELD(stream, pipeline_stage)
  READ_END(stream, NodeDeviceConfigurationProto)
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

void TensorProto::Segment::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, begin)
  WRITE_FIELD(stream, end)
}

void TensorProto::Segment::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TensorProto::Segment)
  READ_FIELD(stream, begin)
  READ_FIELD(stream, end)
  READ_END(stream, TensorProto::Segment)
}

void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dims)
  WRITE_ENUM_FIELD(stream, data_type)
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, raw_data)
  WRITE_FIELD(stream, doc_string)
  WRITE_REPEATED_FIELD(stream, external_data)
  WRITE_REPEATED_FIELD(stream, metadata_props)
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

void TypeProto::Tensor::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, shape)
}

void TypeProto::Tensor::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto::Tensor)
  READ_FIELD(stream, elem_type)
  READ_OPTIONAL_PROTO_FIELD(stream, shape)
  READ_END(stream, TypeProto::Tensor)
}

void TypeProto::SparseTensor::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, shape)
}

void TypeProto::SparseTensor::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto::SparseTensor)
  READ_FIELD(stream, elem_type)
  READ_OPTIONAL_PROTO_FIELD(stream, shape)
  READ_END(stream, TypeProto::SparseTensor)
}

void TypeProto::Sequence::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, elem_type)
}

void TypeProto::Sequence::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto::Sequence)
  READ_OPTIONAL_PROTO_FIELD(stream, elem_type)
  READ_END(stream, TypeProto::Sequence)
}

void TypeProto::Map::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, value_type)
}

void TypeProto::Map::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto::Map)
  READ_FIELD(stream, key_type)
  READ_OPTIONAL_PROTO_FIELD(stream, value_type)
  READ_END(stream, TypeProto::Map)
}

void TypeProto::Optional::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, elem_type)
}

void TypeProto::Optional::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto::Optional)
  READ_OPTIONAL_PROTO_FIELD(stream, elem_type)
  READ_END(stream, TypeProto::Optional)
}

void TypeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, sequence_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, map_type)
  WRITE_FIELD(stream, denotation)
  WRITE_OPTIONAL_PROTO_FIELD(stream, sparse_tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, optional_type)
}

void TypeProto::ParseFromStream(utils::BinaryStream &stream) {
  READ_BEGIN(stream, TypeProto)
  READ_OPTIONAL_PROTO_FIELD(stream, tensor_type)
  READ_OPTIONAL_PROTO_FIELD(stream, sequence_type)
  READ_FIELD(stream, denotation)
  READ_OPTIONAL_PROTO_FIELD(stream, sparse_tensor_type)
  READ_OPTIONAL_PROTO_FIELD(stream, optional_type)
  READ_END(stream, TypeProto)
}

} // namespace onnx2
