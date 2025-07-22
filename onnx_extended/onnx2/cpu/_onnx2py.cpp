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
      [](const onnx2::cls &self) -> py::object {                                               \
        if (self.name##_.has_value())                                                          \
          return py::cast(*self.name##_);                                                      \
        return py::none();                                                                     \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (obj.is_none()) {                                                                   \
          self.name##_.reset();                                                                \
        } else if (py::isinstance<py::int_>(obj)) {                                            \
          self.name##_ = obj.cast<int>();                                                      \
        } else {                                                                               \
          EXT_ENFORCE("unable to set " #name " for class " #cls)                               \
        }                                                                                      \
      },                                                                                       \
      #name)                                                                                   \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name " has a value")

template <typename T> void bind_repeated_field(py::module_ &m, const std::string &name) {
  py::class_<onnx2::utils::RepeatedField<T>>(m, name.c_str(), "repeated field")
      .def(py::init<>())
      .def_readwrite("values", &onnx2::utils::RepeatedField<T>::values)
      .def("add", &onnx2::utils::RepeatedField<T>::add, "adds an empty element")
      .def("clear", &onnx2::utils::RepeatedField<T>::clear, "removes every element")
      .def("__len__", &onnx2::utils::RepeatedField<T>::size, "returns the length")
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
          "extends the list of values")
      .def(
          "__iter__",
          [](onnx2::utils::RepeatedField<T> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>());
}

template <typename T> void bind_optional_field(py::module_ &m, const std::string &name) {
  py::class_<onnx2::utils::OptionalField<T>>(m, name.c_str(), "optional field")
      .def(py::init<>())
      .def_readwrite("value", &onnx2::utils::OptionalField<T>::value)
      .def("__bool__",
           [](const onnx2::utils::OptionalField<T> &self) { return self.has_value(); })
      .def("__eq__", [](const onnx2::utils::OptionalField<T> &self, py::object obj) {
        if (py::isinstance<onnx2::utils::OptionalField<T>>(obj))
          return self == obj.cast<onnx2::utils::OptionalField<T>>();
        return self == obj.cast<T>();
      });
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

  bind_optional_field<int32_t>(m, "OptionalInt32");
  bind_optional_field<int64_t>(m, "OptionalInt64");
  bind_optional_field<uint64_t>(m, "OptionalUInt64");

  py::enum_<onnx2::OperatorStatus>(m, "OperatorStatus", py::arithmetic())
      .value("EXPERIMENTAL", onnx2::OperatorStatus::EXPERIMENTAL)
      .value("STABLE", onnx2::OperatorStatus::STABLE)
      .export_values();

  py::class_<onnx2::StringStringEntryProto>(m, "StringStringEntryProto",
                                            "StringStringEntryProto, a key, a value")
      .def(py::init<>())
      .FIELD(StringStringEntryProto, key)
      .FIELD(StringStringEntryProto, value)
      .ADD_PROTO_SERIALIZATION(StringStringEntryProto);

  bind_repeated_field<onnx2::StringStringEntryProto>(m, "RepeatedFieldStringStringEntryProto");

  py::class_<onnx2::OperatorSetIdProto>(m, "OperatorSetIdProto",
                                        "OperatorSetIdProto, opset definition")
      .def(py::init<>())
      .FIELD(OperatorSetIdProto, domain)
      .FIELD(OperatorSetIdProto, version)
      .ADD_PROTO_SERIALIZATION(OperatorSetIdProto);

  bind_repeated_field<onnx2::OperatorSetIdProto>(m, "RepeatedFieldOperatorSetIdProto");

  py::class_<onnx2::TensorAnnotation>(m, "TensorAnnotation",
                                      "TensorAnnotation, tensor annotation")
      .def(py::init<>())
      .FIELD(TensorAnnotation, tensor_name)
      .FIELD(TensorAnnotation, quant_parameter_tensor_names)
      .ADD_PROTO_SERIALIZATION(TensorAnnotation);

  py::class_<onnx2::IntIntListEntryProto>(m, "IntIntListEntryProto",
                                          "IntIntListEntryProto, tensor annotation")
      .def(py::init<>())
      .FIELD(IntIntListEntryProto, key)
      .FIELD(IntIntListEntryProto, value)
      .ADD_PROTO_SERIALIZATION(IntIntListEntryProto);

  bind_repeated_field<onnx2::IntIntListEntryProto>(m, "RepeatedFieldIntIntListEntryProto");

  py::class_<onnx2::DeviceConfigurationProto>(m, "DeviceConfigurationProto",
                                              "DeviceConfigurationProto")
      .def(py::init<>())
      .FIELD(DeviceConfigurationProto, name)
      .FIELD(DeviceConfigurationProto, num_devices)
      .FIELD(DeviceConfigurationProto, device)
      .ADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  py::class_<onnx2::SimpleShardedDimProto>(m, "SimpleShardedDimProto", "SimpleShardedDimProto")
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

  py::class_<onnx2::NodeDeviceConfigurationProto>(
      m, "NodeDeviceConfigurationProto", "ShardingSpecNodeDeviceConfigurationProtoProto")
      .def(py::init<>())
      .FIELD(NodeDeviceConfigurationProto, configuration_id)
      .FIELD(NodeDeviceConfigurationProto, sharding_spec)
      .FIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage)
      .ADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  py::class_<onnx2::TensorShapeProto::Dimension>(m, "Dimension",
                                                 "Dimension, an integer value or a string")
      .def(py::init<>())
      .FIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .FIELD(TensorShapeProto::Dimension, dim_param)
      .FIELD(TensorShapeProto::Dimension, denotation)
      .ADD_PROTO_SERIALIZATION(TensorShapeProto::Dimension);

  bind_repeated_field<onnx2::TensorShapeProto::Dimension>(m, "RepeatedFieldDimension");

  py::class_<onnx2::TensorShapeProto>(m, "TensorShapeProto",
                                      "TensorShapeProto, multiple DimProto")
      .def(py::init<>())
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

  py::class_<onnx2::TensorProto>(m, "TensorProto", "TensorProto")
      .def(py::init<>())
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

  py::class_<onnx2::SparseTensorProto>(m, "SparseTensorProto",
                                       "SparseTensorProto, sparse tensor")
      .def(py::init<>())
      .FIELD(SparseTensorProto, values)
      .FIELD(SparseTensorProto, indices)
      .FIELD(SparseTensorProto, dims)
      .ADD_PROTO_SERIALIZATION(SparseTensorProto);
}
