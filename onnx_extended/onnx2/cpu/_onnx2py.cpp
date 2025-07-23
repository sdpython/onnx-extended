#include "onnx2.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define ADD_PROTO_SERIALIZATION(cls)                                                           \
  def(                                                                                         \
      "ParseFromString",                                                                       \
      [](onnx2::cls &self, py::bytes data) {                                                   \
        std::string raw = data;                                                                \
        self.ParseFromString(raw);                                                             \
      },                                                                                       \
      "Parses a sequence of bytes to fill this instance")                                      \
      .def(                                                                                    \
          "SerializeToString",                                                                 \
          [](onnx2::cls &self) {                                                               \
            std::string out;                                                                   \
            self.SerializeToString(out);                                                       \
            return py::bytes(out);                                                             \
          },                                                                                   \
          "Serialize into a sequence of bytes.")

#define FIELD(cls, name)                                                                       \
  def_readwrite(#name, &onnx2::cls::name##_, #name)                                            \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name " has a value")

#define FIELD_OPTIONAL_INT(cls, name)                                                          \
  def_property(                                                                                \
      #name,                                                                                   \
      [](onnx2::cls &self) -> py::object {                                                     \
        if (!self.name##_.has_value())                                                         \
          return py::none();                                                                   \
        return py::cast(*self.name##_, py::return_value_policy::reference);                    \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (obj.is_none()) {                                                                   \
          self.name##_.reset();                                                                \
        } else if (py::isinstance<py::int_>(obj)) {                                            \
          self.name##_ = obj.cast<int>();                                                      \
        } else {                                                                               \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'");  \
        }                                                                                      \
      },                                                                                       \
      #name)                                                                                   \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value")

#define FIELD_OPTIONAL_PROTO(cls, name)                                                        \
  def_property(                                                                                \
      #name,                                                                                   \
      [](onnx2::cls &self) -> py::object {                                                     \
        if (!self.name##_.has_value())                                                         \
          return py::none();                                                                   \
        return py::cast(self.name##_.value, py::return_value_policy::reference);               \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (obj.is_none()) {                                                                   \
          self.name##_.reset();                                                                \
        } else if (py::isinstance<onnx2::cls::name##_t>(obj)) {                                \
          self.name##_ = obj.cast<onnx2::cls::name##_t>();                                     \
        } else {                                                                               \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'");  \
        }                                                                                      \
      },                                                                                       \
      #name)                                                                                   \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value")          \
      .def(                                                                                    \
          "add_" #name, [](onnx2::cls & self) -> onnx2::cls::name##_t & {                      \
            self.name##_.set_empty_value();                                                    \
            return *self.name##_;                                                              \
          },                                                                                   \
          py::return_value_policy::reference, "sets an empty value")

#define SHORTEN_CODE(dtype)                                                                    \
  def_property_readonly_static(#dtype, [](py::object) -> int {                                 \
    return static_cast<int>(onnx2::TensorProto::DataType::dtype);                              \
  })

template <typename T> void bind_repeated_field(py::module_ &m, const std::string &name) {
  py::class_<onnx2::utils::RepeatedField<T>>(m, name.c_str(), "repeated field")
      .def(py::init<>())
      .def_readwrite("values", &onnx2::utils::RepeatedField<T>::values)
      .def("add", &onnx2::utils::RepeatedField<T>::add, py::return_value_policy::reference,
           "adds an empty element")
      .def("clear", &onnx2::utils::RepeatedField<T>::clear, "removes every element")
      .def("__len__", &onnx2::utils::RepeatedField<T>::size, "returns the length")
      .def(
          "__getitem__",
          [](onnx2::utils::RepeatedField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self.values[index];
          },
          py::return_value_policy::reference, py::arg("index"),
          "returns the element at position index")
      .def(
          "__delitem__",
          [](onnx2::utils::RepeatedField<T> &self, py::slice slice) {
            size_t start, stop, step, slicelength;
            if (slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
              self.remove_range(start, stop, step);
            }
          },
          "removes elements")
      .def(
          "extend",
          [](onnx2::utils::RepeatedField<T> &self, py::iterable iterable) {
            if (py::isinstance<onnx2::utils::RepeatedField<T>>(iterable)) {
              self.extend(iterable.cast<onnx2::utils::RepeatedField<T>>());
            } else {
              self.extend(iterable.cast<std::vector<T>>());
            }
          },
          py::arg("sequence"), "extends the list of values")
      .def(
          "__iter__",
          [](onnx2::utils::RepeatedField<T> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>());
}

PYBIND11_MODULE(_onnx2py, m) {
  m.doc() =
#if defined(__APPLE__)
      "onnx from python without protobuf"
#else
      R"pbdoc(onnx from python without protobuf)pbdoc"
#endif
      ;

  m.def(
      "utils_onnx2_read_varint64",
      [](py::bytes data) -> py::tuple {
        std::string raw = data;
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
        onnx2::utils::StringStream st(ptr, raw.size());
        int64_t value = st.next_int64();
        return py::make_tuple(value, st.tell());
      },
      py::arg("data"),
      R"pbdoc(Reads a int64_t (protobuf format)

:param data: bytes
:return: 2-tuple, value and number of read bytes
)pbdoc");

  bind_repeated_field<int64_t>(m, "RepeatedFieldInt64");
  bind_repeated_field<int32_t>(m, "RepeatedFieldInt32");
  bind_repeated_field<uint64_t>(m, "RepeatedFieldUInt64");
  bind_repeated_field<float>(m, "RepeatedFieldFloat");
  bind_repeated_field<double>(m, "RepeatedFieldDouble");
  bind_repeated_field<std::string>(m, "RepeatedFieldString");

  py::class_<onnx2::Message>(m, "Message", "Message, base class for all onnx2 classes")
      .def(py::init<>());

  py::enum_<onnx2::OperatorStatus>(m, "OperatorStatus", py::arithmetic())
      .value("EXPERIMENTAL", onnx2::OperatorStatus::EXPERIMENTAL)
      .value("STABLE", onnx2::OperatorStatus::STABLE)
      .export_values();

  py::class_<onnx2::StringStringEntryProto, onnx2::Message>(
      m, "StringStringEntryProto", "StringStringEntryProto, a key, a value")
      .def(py::init<>())
      .FIELD(StringStringEntryProto, key)
      .FIELD(StringStringEntryProto, value)
      .ADD_PROTO_SERIALIZATION(StringStringEntryProto);

  bind_repeated_field<onnx2::StringStringEntryProto>(m, "RepeatedFieldStringStringEntryProto");

  py::class_<onnx2::OperatorSetIdProto, onnx2::Message>(m, "OperatorSetIdProto",
                                                        "OperatorSetIdProto, opset definition")
      .def(py::init<>())
      .FIELD(OperatorSetIdProto, domain)
      .FIELD(OperatorSetIdProto, version)
      .ADD_PROTO_SERIALIZATION(OperatorSetIdProto);

  bind_repeated_field<onnx2::OperatorSetIdProto>(m, "RepeatedFieldOperatorSetIdProto");

  py::class_<onnx2::TensorAnnotation, onnx2::Message>(m, "TensorAnnotation",
                                                      "TensorAnnotation, tensor annotation")
      .def(py::init<>())
      .FIELD(TensorAnnotation, tensor_name)
      .FIELD(TensorAnnotation, quant_parameter_tensor_names)
      .ADD_PROTO_SERIALIZATION(TensorAnnotation);

  py::class_<onnx2::IntIntListEntryProto, onnx2::Message>(
      m, "IntIntListEntryProto", "IntIntListEntryProto, tensor annotation")
      .def(py::init<>())
      .FIELD(IntIntListEntryProto, key)
      .FIELD(IntIntListEntryProto, value)
      .ADD_PROTO_SERIALIZATION(IntIntListEntryProto);

  bind_repeated_field<onnx2::IntIntListEntryProto>(m, "RepeatedFieldIntIntListEntryProto");

  py::class_<onnx2::DeviceConfigurationProto, onnx2::Message>(m, "DeviceConfigurationProto",
                                                              "DeviceConfigurationProto")
      .def(py::init<>())
      .FIELD(DeviceConfigurationProto, name)
      .FIELD(DeviceConfigurationProto, num_devices)
      .FIELD(DeviceConfigurationProto, device)
      .ADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  py::class_<onnx2::SimpleShardedDimProto, onnx2::Message>(m, "SimpleShardedDimProto",
                                                           "SimpleShardedDimProto")
      .def(py::init<>())
      .FIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .FIELD(SimpleShardedDimProto, dim_param)
      .FIELD(SimpleShardedDimProto, num_shards)
      .ADD_PROTO_SERIALIZATION(SimpleShardedDimProto);

  bind_repeated_field<onnx2::SimpleShardedDimProto>(m, "RepeatedFieldSimpleShardedDimProto");

  py::class_<onnx2::ShardedDimProto>(m, "ShardedDimProto", "ShardedDimProto")
      .def(py::init<>())
      .FIELD(ShardedDimProto, axis)
      .FIELD(ShardedDimProto, simple_sharding)
      .ADD_PROTO_SERIALIZATION(ShardedDimProto);

  bind_repeated_field<onnx2::ShardedDimProto>(m, "RepeatedFieldShardedDimProto");

  py::class_<onnx2::ShardingSpecProto>(m, "ShardingSpecProto", "ShardingSpecProto")
      .def(py::init<>())
      .FIELD(ShardingSpecProto, tensor_name)
      .FIELD(ShardingSpecProto, device)
      .FIELD(ShardingSpecProto, index_to_device_group_map)
      .FIELD(ShardingSpecProto, sharded_dim)
      .ADD_PROTO_SERIALIZATION(ShardingSpecProto);

  bind_repeated_field<onnx2::ShardingSpecProto>(m, "RepeatedFieldShardingSpecProto");

  py::class_<onnx2::NodeDeviceConfigurationProto, onnx2::Message>(
      m, "NodeDeviceConfigurationProto", "ShardingSpecNodeDeviceConfigurationProtoProto")
      .def(py::init<>())
      .FIELD(NodeDeviceConfigurationProto, configuration_id)
      .FIELD(NodeDeviceConfigurationProto, sharding_spec)
      .FIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage)
      .ADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  py::class_<onnx2::TensorShapeProto, onnx2::Message> cls_tensor_shape_proto(
      m, "TensorShapeProto", "TensorShapeProto");

  py::class_<onnx2::TensorShapeProto::Dimension, onnx2::Message>(
      cls_tensor_shape_proto, "Dimension", "Dimension, an integer value or a string")
      .def(py::init<>())
      .FIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .FIELD(TensorShapeProto::Dimension, dim_param)
      .FIELD(TensorShapeProto::Dimension, denotation)
      .ADD_PROTO_SERIALIZATION(TensorShapeProto::Dimension);

  bind_repeated_field<onnx2::TensorShapeProto::Dimension>(m, "RepeatedFieldDimension");

  cls_tensor_shape_proto.def(py::init<>())
      .FIELD(TensorShapeProto, dim)
      .ADD_PROTO_SERIALIZATION(TensorShapeProto);

  py::enum_<onnx2::TensorProto::DataType>(m, "DataType", py::arithmetic())
      .value("UNDEFINED", onnx2::TensorProto::DataType::UNDEFINED)
      .value("FLOAT", onnx2::TensorProto::DataType::FLOAT)
      .value("UINT8", onnx2::TensorProto::DataType::UINT8)
      .value("INT8", onnx2::TensorProto::DataType::INT8)
      .value("UINT16", onnx2::TensorProto::DataType::UINT16)
      .value("INT16", onnx2::TensorProto::DataType::INT16)
      .value("INT32", onnx2::TensorProto::DataType::INT32)
      .value("INT64", onnx2::TensorProto::DataType::INT64)
      .value("STRING", onnx2::TensorProto::DataType::STRING)
      .value("BOOL", onnx2::TensorProto::DataType::BOOL)
      .value("FLOAT16", onnx2::TensorProto::DataType::FLOAT16)
      .value("DOUBLE", onnx2::TensorProto::DataType::DOUBLE)
      .value("UINT32", onnx2::TensorProto::DataType::UINT32)
      .value("UINT64", onnx2::TensorProto::DataType::UINT64)
      .value("COMPLEX64", onnx2::TensorProto::DataType::COMPLEX64)
      .value("COMPLEX128", onnx2::TensorProto::DataType::COMPLEX128)
      .value("BFLOAT16", onnx2::TensorProto::DataType::BFLOAT16)
      .value("FLOAT8E4M3FN", onnx2::TensorProto::DataType::FLOAT8E4M3FN)
      .value("FLOAT8E4M3FNUZ", onnx2::TensorProto::DataType::FLOAT8E4M3FNUZ)
      .value("FLOAT8E5M2", onnx2::TensorProto::DataType::FLOAT8E5M2)
      .value("FLOAT8E5M2FNUZ", onnx2::TensorProto::DataType::FLOAT8E5M2FNUZ)
      .value("UINT4", onnx2::TensorProto::DataType::UINT4)
      .value("INT4", onnx2::TensorProto::DataType::INT4)
      .value("FLOAT4E2M1", onnx2::TensorProto::DataType::FLOAT4E2M1)
      .value("FLOAT8E8M0", onnx2::TensorProto::DataType::FLOAT8E8M0)
      .export_values();

  py::class_<onnx2::TensorProto, onnx2::Message>(m, "TensorProto", "TensorProto")
      .def(py::init<>())
      .SHORTEN_CODE(UNDEFINED)
      .SHORTEN_CODE(FLOAT)
      .SHORTEN_CODE(UINT8)
      .SHORTEN_CODE(INT8)
      .SHORTEN_CODE(UINT16)
      .SHORTEN_CODE(INT16)
      .SHORTEN_CODE(INT32)
      .SHORTEN_CODE(INT64)
      .SHORTEN_CODE(STRING)
      .SHORTEN_CODE(BOOL)
      .SHORTEN_CODE(FLOAT16)
      .SHORTEN_CODE(DOUBLE)
      .SHORTEN_CODE(UINT32)
      .SHORTEN_CODE(UINT64)
      .SHORTEN_CODE(COMPLEX64)
      .SHORTEN_CODE(COMPLEX128)
      .SHORTEN_CODE(BFLOAT16)
      .SHORTEN_CODE(FLOAT8E4M3FN)
      .SHORTEN_CODE(FLOAT8E4M3FNUZ)
      .SHORTEN_CODE(FLOAT8E5M2)
      .SHORTEN_CODE(FLOAT8E5M2FNUZ)
      .SHORTEN_CODE(UINT4)
      .SHORTEN_CODE(INT4)
      .SHORTEN_CODE(FLOAT4E2M1)
      .SHORTEN_CODE(FLOAT8E8M0)
      .FIELD(TensorProto, dims)
      .def_property(
          "data_type",
          [](const onnx2::TensorProto &self) -> onnx2::TensorProto::DataType {
            return self.data_type_;
          },
          [](onnx2::TensorProto &self, py::object obj) {
            if (py::isinstance<py::int_>(obj)) {
              self.data_type_ = static_cast<onnx2::TensorProto::DataType>(obj.cast<int>());
            } else {
              self.data_type_ = obj.cast<onnx2::TensorProto::DataType>();
            }
          },
          "data_type")
      .FIELD(TensorProto, name)
      .FIELD(TensorProto, doc_string)
      .FIELD(TensorProto, external_data)
      .FIELD(TensorProto, metadata_props)
      .FIELD(TensorProto, dims)
      .FIELD(TensorProto, double_data)
      .FIELD(TensorProto, float_data)
      .FIELD(TensorProto, int64_data)
      .FIELD(TensorProto, int32_data)
      .FIELD(TensorProto, uint64_data)
      .def_property(
          "string_data",
          [](const onnx2::TensorProto &self) -> py::list {
            py::list result;
            for (const auto &s : self.string_data_) {
              result.append(py::bytes(s));
            }
            return result;
          },
          [](onnx2::TensorProto &self, py::list data) {
            self.string_data_.reserve(py::len(data));

            for (const auto &item : data) {
              if (py::isinstance<py::bytes>(item)) {
                self.string_data_.emplace_back(item.cast<std::string>());
              } else if (py::isinstance<py::str>(item)) {
                self.string_data_.emplace_back(item.cast<std::string>());
              } else {
                EXT_THROW("unable to convert one item from the list into a string")
              }
            }
          },
          "string_data")
      .def_property(
          "raw_data",
          [](const onnx2::TensorProto &self) -> py::bytes {
            return py::bytes(reinterpret_cast<const char *>(self.raw_data_.data()),
                             self.raw_data_.size());
          },
          [](onnx2::TensorProto &self, py::bytes data) {
            std::string raw = data;
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            self.raw_data_.resize(raw.size());
            memcpy(self.raw_data_.data(), ptr, raw.size());
          },
          "raw_data")
      .ADD_PROTO_SERIALIZATION(TensorProto);

  py::class_<onnx2::SparseTensorProto, onnx2::Message>(m, "SparseTensorProto",
                                                       "SparseTensorProto, sparse tensor")
      .def(py::init<>())
      .FIELD(SparseTensorProto, values)
      .FIELD(SparseTensorProto, indices)
      .FIELD(SparseTensorProto, dims)
      .ADD_PROTO_SERIALIZATION(SparseTensorProto);

  py::class_<onnx2::TypeProto, onnx2::Message> cls_type_proto(m, "TypeProto", "TypeProto");

  py::class_<onnx2::TypeProto::Tensor, onnx2::Message>(cls_type_proto, "Tensor",
                                                       "Tensor, nested class of TypeProto")
      .def(py::init<>())
      .FIELD_OPTIONAL_INT(TypeProto::Tensor, elem_type)
      .FIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape)
      .ADD_PROTO_SERIALIZATION(TypeProto::Tensor);

  py::class_<onnx2::TypeProto::SparseTensor, onnx2::Message>(
      cls_type_proto, "SparseTensor", "SparseTensor, nested class of TypeProto")
      .def(py::init<>())
      .FIELD_OPTIONAL_INT(TypeProto::SparseTensor, elem_type)
      .FIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape)
      .ADD_PROTO_SERIALIZATION(TypeProto::SparseTensor);

  py::class_<onnx2::TypeProto::Sequence, onnx2::Message>(cls_type_proto, "Sequence",
                                                         "Sequence, nested class of TypeProto")
      .def(py::init<>())
      .FIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type)
      .ADD_PROTO_SERIALIZATION(TypeProto::Sequence);

  py::class_<onnx2::TypeProto::Optional, onnx2::Message>(cls_type_proto, "Optional",
                                                         "Optional, nested class of TypeProto")
      .def(py::init<>())
      .FIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type)
      .ADD_PROTO_SERIALIZATION(TypeProto::Optional);

  py::class_<onnx2::TypeProto::Map, onnx2::Message>(cls_type_proto, "Map",
                                                    "Map, nested class of TypeProto")
      .def(py::init<>())
      .FIELD(TypeProto::Map, key_type)
      .FIELD_OPTIONAL_PROTO(TypeProto::Map, value_type)
      .ADD_PROTO_SERIALIZATION(TypeProto::Map);

  cls_type_proto.def(py::init<>())
      .FIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .FIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .FIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .FIELD(TypeProto, denotation)
      .FIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .FIELD_OPTIONAL_PROTO(TypeProto, optional_type)
      .ADD_PROTO_SERIALIZATION(TypeProto);
}
