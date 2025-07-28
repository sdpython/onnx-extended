#include "onnx2.h"
#include "stream_class.hpp"
#include <sstream>

namespace onnx2 {

// StringStringEntryProto
IMPLEMENT_PROTO(StringStringEntryProto)

void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_FIELD(stream, value)
}

void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, StringStringEntryProto) //
    READ_FIELD(stream, key)                    //
    READ_FIELD(stream, value)                  //
    READ_END(stream, StringStringEntryProto)   //  // NOLINT
}

std::vector<std::string> StringStringEntryProto::SerializeToVectorString() const {
  return {write_as_string(NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value))};
}

// TensorAnnotation
IMPLEMENT_PROTO(TensorAnnotation)

void TensorAnnotation::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, tensor_name)
  WRITE_REPEATED_FIELD(stream, quant_parameter_tensor_names)
}

void TensorAnnotation::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorAnnotation)                      //
    READ_FIELD(stream, tensor_name)                           //
    READ_REPEATED_FIELD(stream, quant_parameter_tensor_names) //
    READ_END(stream, TensorAnnotation)                        //  // NOLINT
}

std::vector<std::string> TensorAnnotation::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(tensor_name),
                                        NAME_EXIST_VALUE(quant_parameter_tensor_names));
}

// IntIntListEntryProto
IMPLEMENT_PROTO(IntIntListEntryProto)

void IntIntListEntryProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key)
  WRITE_REPEATED_FIELD(stream, value)
}

void IntIntListEntryProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, IntIntListEntryProto) //
    READ_FIELD(stream, key)                  //
    READ_REPEATED_FIELD(stream, value)       //
    READ_END(stream, IntIntListEntryProto)   //
}

std::vector<std::string> IntIntListEntryProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value));
}

// DeviceConfigurationProto
IMPLEMENT_PROTO(DeviceConfigurationProto)

void DeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, num_devices)
  WRITE_REPEATED_FIELD(stream, device)
}

void DeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, DeviceConfigurationProto) //
    READ_FIELD(stream, name)                     //
    READ_FIELD(stream, num_devices)              //
    READ_REPEATED_FIELD(stream, device)          //
    READ_END(stream, DeviceConfigurationProto)   //
}

std::vector<std::string> DeviceConfigurationProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(num_devices),
                                        NAME_EXIST_VALUE(device));
}

// SimpleShardedDimProto
IMPLEMENT_PROTO(SimpleShardedDimProto)

void SimpleShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, dim_value)
  WRITE_FIELD(stream, dim_param)
  WRITE_FIELD(stream, num_shards)
}

void SimpleShardedDimProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, SimpleShardedDimProto) //
    READ_FIELD(stream, dim_value)             //
    READ_FIELD(stream, dim_param)             //
    READ_FIELD(stream, num_shards)            //
    READ_END(stream, SimpleShardedDimProto)   //
}

std::vector<std::string> SimpleShardedDimProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(dim_value), NAME_EXIST_VALUE(dim_param),
                                        NAME_EXIST_VALUE(num_shards));
}

// ShardedDimProto
IMPLEMENT_PROTO(ShardedDimProto)

void ShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, axis)
  WRITE_REPEATED_FIELD(stream, simple_sharding)
}

void ShardedDimProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, ShardedDimProto)          //
    READ_FIELD(stream, axis)                     //
    READ_REPEATED_FIELD(stream, simple_sharding) //
    READ_END(stream, ShardedDimProto)            //
}

std::vector<std::string> ShardedDimProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(axis), NAME_EXIST_VALUE(simple_sharding));
}

// ShardingSpecProto
IMPLEMENT_PROTO(ShardingSpecProto)

void ShardingSpecProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, tensor_name)
  WRITE_REPEATED_FIELD(stream, device)
  WRITE_REPEATED_FIELD(stream, index_to_device_group_map)
  WRITE_REPEATED_FIELD(stream, sharded_dim)
}

void ShardingSpecProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, ShardingSpecProto)                  //
    READ_FIELD(stream, tensor_name)                        //
    READ_REPEATED_FIELD(stream, device)                    //
    READ_REPEATED_FIELD(stream, index_to_device_group_map) //
    READ_REPEATED_FIELD(stream, sharded_dim)               //
    READ_END(stream, ShardingSpecProto)                    //  // NOLINT
}

std::vector<std::string> ShardingSpecProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(tensor_name), NAME_EXIST_VALUE(device),
                                        NAME_EXIST_VALUE(index_to_device_group_map),
                                        NAME_EXIST_VALUE(sharded_dim));
}

// NodeDeviceConfigurationProto
IMPLEMENT_PROTO(NodeDeviceConfigurationProto)

void NodeDeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, configuration_id)
  WRITE_REPEATED_FIELD(stream, sharding_spec)
  WRITE_FIELD(stream, pipeline_stage)
}

void NodeDeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, NodeDeviceConfigurationProto) //
    READ_FIELD(stream, configuration_id)             //
    READ_REPEATED_FIELD(stream, sharding_spec)       //
    READ_FIELD(stream, pipeline_stage)               //
    READ_END(stream, NodeDeviceConfigurationProto)   //
}

std::vector<std::string> NodeDeviceConfigurationProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(configuration_id),
                                        NAME_EXIST_VALUE(sharding_spec),
                                        NAME_EXIST_VALUE(pipeline_stage));
}

// OperatorSetIdProto
IMPLEMENT_PROTO(OperatorSetIdProto)

void OperatorSetIdProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, domain)
  WRITE_FIELD(stream, version)
}

void OperatorSetIdProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, OperatorSetIdProto) //
    READ_FIELD(stream, domain)             //
    READ_FIELD(stream, version)            //
    READ_END(stream, OperatorSetIdProto)   //
}

std::vector<std::string> OperatorSetIdProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(version));
}

// TensorShapeProto::Dimension
IMPLEMENT_PROTO(TensorShapeProto::Dimension)

void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, dim_value)
  WRITE_FIELD(stream, dim_param)
  WRITE_FIELD(stream, denotation)
}

void TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorShapeProto::Dimension) //
    READ_FIELD(stream, dim_value)                   //
    READ_FIELD(stream, dim_param)                   //
    READ_FIELD(stream, denotation)                  //
    READ_END(stream, TensorShapeProto::Dimension)   //
}

std::vector<std::string> TensorShapeProto::Dimension::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(dim_value), NAME_EXIST_VALUE(dim_param),
                                        NAME_EXIST_VALUE(denotation));
}

// TensorShapeProto
IMPLEMENT_PROTO(TensorShapeProto)

void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_REPEATED_FIELD(stream, dim)
}

void TensorShapeProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorShapeProto) //
    READ_REPEATED_FIELD(stream, dim)     //
    READ_END(stream, TensorShapeProto)   //
}

std::vector<std::string> TensorShapeProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(dim));
}

// TensorProto::Segment
IMPLEMENT_PROTO(TensorProto::Segment)

void TensorProto::Segment::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, begin)
  WRITE_FIELD(stream, end)
}

void TensorProto::Segment::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorProto::Segment) //
    READ_FIELD(stream, begin)                //
    READ_FIELD(stream, end)                  //
    READ_END(stream, TensorProto::Segment)   //
}

std::vector<std::string> TensorProto::Segment::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(begin), NAME_EXIST_VALUE(end));
}

// TensorProto
IMPLEMENT_PROTO(TensorProto)

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

void TensorProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TensorProto)             //
    READ_REPEATED_FIELD(stream, dims)           //
    READ_ENUM_FIELD(stream, data_type)          //
    READ_FIELD(stream, name)                    //
    READ_FIELD(stream, doc_string)              //
    READ_FIELD(stream, raw_data)                //
    READ_REPEATED_FIELD(stream, external_data)  //
    READ_REPEATED_FIELD(stream, metadata_props) //
    READ_REPEATED_FIELD(stream, double_data)    //
    READ_REPEATED_FIELD(stream, float_data)     //
    READ_REPEATED_FIELD(stream, int32_data)     //
    READ_REPEATED_FIELD(stream, int64_data)     //
    READ_REPEATED_FIELD(stream, uint64_data)    //
    READ_REPEATED_FIELD(stream, string_data)    //
    READ_END(stream, TensorProto)               //
}

std::vector<std::string> TensorProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(
      NAME_EXIST_VALUE(dims), NAME_EXIST_VALUE(data_type), NAME_EXIST_VALUE(name),
      NAME_EXIST_VALUE(segment), NAME_EXIST_VALUE(raw_data), NAME_EXIST_VALUE(doc_string),
      NAME_EXIST_VALUE(external_data), NAME_EXIST_VALUE(metadata_props), NAME_EXIST_VALUE(double_data),
      NAME_EXIST_VALUE(float_data), NAME_EXIST_VALUE(int32_data), NAME_EXIST_VALUE(int64_data),
      NAME_EXIST_VALUE(uint64_data), NAME_EXIST_VALUE(string_data));
}

// SparseTensorProto
IMPLEMENT_PROTO(SparseTensorProto)

void SparseTensorProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, values)
  WRITE_FIELD(stream, indices)
  WRITE_REPEATED_FIELD(stream, dims)
}

void SparseTensorProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, SparseTensorProto) //
    READ_FIELD(stream, values)            //
    READ_FIELD(stream, indices)           //
    READ_REPEATED_FIELD(stream, dims)     //
    READ_END(stream, SparseTensorProto)   //
}

std::vector<std::string> SparseTensorProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(values), NAME_EXIST_VALUE(indices),
                                        NAME_EXIST_VALUE(dims));
}

// TypeProto::Tensor
IMPLEMENT_PROTO(TypeProto::Tensor)

void TypeProto::Tensor::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, shape)
}

void TypeProto::Tensor::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto::Tensor)    //
    READ_FIELD(stream, elem_type)            //
    READ_OPTIONAL_PROTO_FIELD(stream, shape) //
    READ_END(stream, TypeProto::Tensor)      //
}

std::vector<std::string> TypeProto::Tensor::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::SparseTensor
IMPLEMENT_PROTO(TypeProto::SparseTensor)

void TypeProto::SparseTensor::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, shape)
}

void TypeProto::SparseTensor::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto::SparseTensor) //
    READ_FIELD(stream, elem_type)               //
    READ_OPTIONAL_PROTO_FIELD(stream, shape)    //
    READ_END(stream, TypeProto::SparseTensor)   //
}

std::vector<std::string> TypeProto::SparseTensor::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::Sequence
IMPLEMENT_PROTO(TypeProto::Sequence)

void TypeProto::Sequence::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, elem_type)
}

void TypeProto::Sequence::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto::Sequence)      //
    READ_OPTIONAL_PROTO_FIELD(stream, elem_type) //
    READ_END(stream, TypeProto::Sequence)        //
}

std::vector<std::string> TypeProto::Sequence::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(elem_type));
}

//  TypeProto::Map
IMPLEMENT_PROTO(TypeProto::Map)

void TypeProto::Map::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, key_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, value_type)
}

void TypeProto::Map::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto::Map)            //
    READ_FIELD(stream, key_type)                  //
    READ_OPTIONAL_PROTO_FIELD(stream, value_type) //
    READ_END(stream, TypeProto::Map)              //
}

std::vector<std::string> TypeProto::Map::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(key_type), NAME_EXIST_VALUE(value_type));
}

// TypeProto::Optional
IMPLEMENT_PROTO(TypeProto::Optional)

void TypeProto::Optional::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, elem_type)
}

void TypeProto::Optional::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto::Optional)      //
    READ_OPTIONAL_PROTO_FIELD(stream, elem_type) //
    READ_END(stream, TypeProto::Optional)        //
}

std::vector<std::string> TypeProto::Optional::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(elem_type));
}

// TypeProto
IMPLEMENT_PROTO(TypeProto)

void TypeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_OPTIONAL_PROTO_FIELD(stream, tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, sequence_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, map_type)
  WRITE_FIELD(stream, denotation)
  WRITE_OPTIONAL_PROTO_FIELD(stream, sparse_tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(stream, optional_type)
}

void TypeProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, TypeProto)                         //
    READ_OPTIONAL_PROTO_FIELD(stream, tensor_type)        //
    READ_OPTIONAL_PROTO_FIELD(stream, sequence_type)      //
    READ_FIELD(stream, denotation)                        //
    READ_OPTIONAL_PROTO_FIELD(stream, sparse_tensor_type) //
    READ_OPTIONAL_PROTO_FIELD(stream, optional_type)      //
    READ_END(stream, TypeProto)                           //
}

std::vector<std::string> TypeProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(tensor_type), NAME_EXIST_VALUE(sequence_type),
                                        NAME_EXIST_VALUE(map_type), NAME_EXIST_VALUE(denotation),
                                        NAME_EXIST_VALUE(sparse_tensor_type),
                                        NAME_EXIST_VALUE(optional_type));
}

// ValueInfoProto
IMPLEMENT_PROTO(ValueInfoProto)

void ValueInfoProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, name)
  WRITE_OPTIONAL_PROTO_FIELD(stream, type)
  WRITE_FIELD(stream, doc_string)
  WRITE_REPEATED_FIELD(stream, metadata_props)
}

void ValueInfoProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, ValueInfoProto)          //
    READ_FIELD(stream, name)                    //
    READ_OPTIONAL_PROTO_FIELD(stream, type)     //
    READ_FIELD(stream, doc_string)              //
    READ_REPEATED_FIELD(stream, metadata_props) //
    READ_END(stream, ValueInfoProto)            //
}

std::vector<std::string> ValueInfoProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(type),
                                        NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(metadata_props));
}

// AttributeProto
IMPLEMENT_PROTO(AttributeProto)

void AttributeProto::SerializeToStream(utils::BinaryWriteStream &stream) const {
  WRITE_FIELD(stream, name)
  WRITE_FIELD(stream, ref_attr_name)
  WRITE_ENUM_FIELD(stream, type)
  WRITE_FIELD(stream, doc_string)
  WRITE_FIELD(stream, f)
  WRITE_FIELD(stream, i)
  WRITE_FIELD(stream, s)
  WRITE_OPTIONAL_PROTO_FIELD(stream, t)
  WRITE_OPTIONAL_PROTO_FIELD(stream, sparse_tensor)
  // WRITE_FIELD(stream, g)
  WRITE_REPEATED_FIELD(stream, floats)
  WRITE_REPEATED_FIELD(stream, ints)
  WRITE_REPEATED_FIELD(stream, strings)
  WRITE_REPEATED_FIELD(stream, tensors)
  WRITE_REPEATED_FIELD(stream, sparse_tensors)
  // WRITE_REPEATED_FIELD(stream, graphs)
}

void AttributeProto::ParseFromStream(utils::BinaryStream &stream){
    READ_BEGIN(stream, AttributeProto)               //
    READ_FIELD(stream, name)                         //
    READ_FIELD(stream, ref_attr_name)                //
    READ_ENUM_FIELD(stream, type)                    //
    READ_FIELD(stream, doc_string)                   //
    READ_FIELD(stream, f)                            //
    READ_FIELD(stream, i)                            //
    READ_FIELD(stream, s)                            //
    READ_OPTIONAL_PROTO_FIELD(stream, t)             //
    READ_OPTIONAL_PROTO_FIELD(stream, sparse_tensor) //
    // WRITE_FIELD(stream, g)
    READ_REPEATED_FIELD(stream, floats)         //
    READ_REPEATED_FIELD(stream, ints)           //
    READ_REPEATED_FIELD(stream, strings)        //
    READ_REPEATED_FIELD(stream, tensors)        //
    READ_REPEATED_FIELD(stream, sparse_tensors) //
    // READ_REPEATED_FIELD(stream, graphs)
    READ_END(stream, AttributeProto) //
}

std::vector<std::string> AttributeProto::SerializeToVectorString() const {
  return write_proto_into_vector_string(
      NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(ref_attr_name), NAME_EXIST_VALUE(doc_string),
      NAME_EXIST_VALUE(type), NAME_EXIST_VALUE(f), NAME_EXIST_VALUE(i), NAME_EXIST_VALUE(s),
      NAME_EXIST_VALUE(t), NAME_EXIST_VALUE(sparse_tensor), NAME_EXIST_VALUE(floats),
      NAME_EXIST_VALUE(ints), NAME_EXIST_VALUE(strings), NAME_EXIST_VALUE(tensors),
      NAME_EXIST_VALUE(sparse_tensors));
}

} // namespace onnx2
