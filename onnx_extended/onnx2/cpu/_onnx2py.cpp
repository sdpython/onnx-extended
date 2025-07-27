#include "onnx2.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define PYDEFINE_PROTO(m, cls)                                                                 \
  py::class_<onnx2::cls, onnx2::Message>(m, #cls, onnx2::cls::DOC).def(py::init<>())

#define PYDEFINE_SUBPROTO(m, cls, subname)                                                     \
  py::class_<onnx2::cls::subname, onnx2::Message>(m, #subname, onnx2::cls::subname::DOC)       \
      .def(py::init<>())

#define PYDEFINE_PROTO_WITH_SUBTYPES(m, cls, name)                                             \
  py::class_<onnx2::cls, onnx2::Message> name(m, #cls, onnx2::cls::DOC);                       \
  name.def(py::init<>());

#define PYADD_PROTO_SERIALIZATION(cls)                                                         \
  def(                                                                                         \
      "ParseFromString",                                                                       \
      [](onnx2::cls &self, py::bytes data) {                                                   \
        std::string raw = data;                                                                \
        self.ParseFromString(raw);                                                             \
      },                                                                                       \
      "Parses a sequence of bytes to fill this instance.")                                     \
      .def(                                                                                    \
          "SerializeToString",                                                                 \
          [](onnx2::cls &self) {                                                               \
            std::string out;                                                                   \
            self.SerializeToString(out);                                                       \
            return py::bytes(out);                                                             \
          },                                                                                   \
          "Serializes this instance into a sequence of bytes.")                                \
      .def(                                                                                    \
          "__str__",                                                                           \
          [](onnx2::cls &self) -> std::string {                                                \
            std::vector<std::string> rows = self.SerializeToVectorString();                    \
            return onnx2::utils::join_string(rows);                                            \
          },                                                                                   \
          "Creates a printable string for this class.")

#define PYFIELD(cls, name)                                                                     \
  def_readwrite(#name, &onnx2::cls::name##_, #name)                                            \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_STR(cls, name)                                                                 \
  def_property(                                                                                \
      #name,                                                                                   \
      [](const onnx2::cls &self) -> std::string {                                              \
        std::string s = self.name().as_string();                                               \
        return s;                                                                              \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (py::isinstance<py::str>(obj)) {                                                    \
          std::string st = obj.cast<std::string>();                                            \
          self.set_##name(st);                                                                 \
        } else {                                                                               \
          self.set_##name(obj.cast<onnx2::cls::name##_t &>());                                 \
        }                                                                                      \
      },                                                                                       \
      onnx2::cls::DOC_##name)                                                                  \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value")

#define PYFIELD_OPTIONAL_INT(cls, name)                                                        \
  def_property(                                                                                \
      #name,                                                                                   \
      [](onnx2::cls &self) -> py::object {                                                     \
        if (!self.has_##name())                                                                \
          return py::none();                                                                   \
        return py::cast(self.name(), py::return_value_policy::reference);                      \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (obj.is_none()) {                                                                   \
          self.reset_##name();                                                                 \
        } else if (py::isinstance<py::int_>(obj)) {                                            \
          self.set_##name(obj.cast<int>());                                                    \
        } else {                                                                               \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'."); \
        }                                                                                      \
      },                                                                                       \
      onnx2::cls::DOC_##name)                                                                  \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                      \
  def_property(                                                                                \
      #name,                                                                                   \
      [](onnx2::cls &self) -> py::object {                                                     \
        if (!self.name##_.has_value()) {                                                       \
          if (self.has_oneof_##name())                                                         \
            return py::none();                                                                 \
          self.name##_.set_empty_value();                                                      \
        }                                                                                      \
        return py::cast(self.name##_.value, py::return_value_policy::reference);               \
      },                                                                                       \
      [](onnx2::cls &self, py::object obj) {                                                   \
        if (obj.is_none()) {                                                                   \
          self.name##_.reset();                                                                \
        } else if (py::isinstance<onnx2::cls::name##_t>(obj)) {                                \
          self.name##_ = obj.cast<onnx2::cls::name##_t &>();                                   \
        } else {                                                                               \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'."); \
        }                                                                                      \
      },                                                                                       \
      onnx2::cls::DOC_##name)                                                                  \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")         \
      .def(                                                                                    \
          "add_" #name, [](onnx2::cls & self) -> onnx2::cls::name##_t & {                      \
            self.name##_.set_empty_value();                                                    \
            return *self.name##_;                                                              \
          },                                                                                   \
          py::return_value_policy::reference, "Sets an empty value.")

#define SHORTEN_CODE(dtype)                                                                    \
  def_property_readonly_static(#dtype, [](py::object) -> int {                                 \
    return static_cast<int>(onnx2::TensorProto::DataType::dtype);                              \
  })

template <typename T> void define_repeated_field_type(py::module_ &m, const std::string &name) {
  py::class_<onnx2::utils::RepeatedField<T>>(m, name.c_str(), "repeated field")
      .def(py::init<>())
      .def_readwrite("values", &onnx2::utils::RepeatedField<T>::values)
      .def("add", &onnx2::utils::RepeatedField<T>::add, py::return_value_policy::reference,
           "Adds an empty element.")
      .def("clear", &onnx2::utils::RepeatedField<T>::clear, "Removes every element.")
      .def("__len__", &onnx2::utils::RepeatedField<T>::size, "Returns the number of elements.")
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
          "Returns the element at position index.")
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
          py::arg("sequence"), "Extends the list of values.")
      .def(
          "__iter__",
          [](onnx2::utils::RepeatedField<T> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Iterates over the elements.");
}

template <>
void define_repeated_field_type<onnx2::utils::String>(py::module_ &m, const std::string &name) {
  py::class_<onnx2::utils::RepeatedField<onnx2::utils::String>>(m, name.c_str(),
                                                                "repeated field")
      .def(py::init<>())
      .def_readwrite("values", &onnx2::utils::RepeatedField<onnx2::utils::String>::values)
      .def("add", &onnx2::utils::RepeatedField<onnx2::utils::String>::add,
           py::return_value_policy::reference, "Adds an empty element.")
      .def("clear", &onnx2::utils::RepeatedField<onnx2::utils::String>::clear,
           "Removes every element.")
      .def("__len__", &onnx2::utils::RepeatedField<onnx2::utils::String>::size,
           "Returns the number of elements.")
      .def(
          "__getitem__",
          [](onnx2::utils::RepeatedField<onnx2::utils::String> &self,
             int index) -> onnx2::utils::String & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self.values[index];
          },
          py::return_value_policy::reference, py::arg("index"),
          "Returns the element at position index.")
      .def(
          "__delitem__",
          [](onnx2::utils::RepeatedField<onnx2::utils::String> &self, py::slice slice) {
            size_t start, stop, step, slicelength;
            if (slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
              self.remove_range(start, stop, step);
            }
          },
          "Removes every element.")
      .def(
          "extend",
          [](onnx2::utils::RepeatedField<onnx2::utils::String> &self, py::iterable iterable) {
            if (py::isinstance<onnx2::utils::RepeatedField<onnx2::utils::String>>(iterable)) {
              self.extend(iterable.cast<onnx2::utils::RepeatedField<onnx2::utils::String> &>());
            } else {
              std::vector<onnx2::utils::String> values;
              for (auto it : iterable) {
                if (py::isinstance<onnx2::utils::String>(it)) {
                  values.push_back(it.cast<onnx2::utils::String &>());
                } else {
                  values.emplace_back(onnx2::utils::String(it.cast<std::string>()));
                }
              }
              self.extend(values);
            }
          },
          py::arg("sequence"), "Extends the list of values.")
      .def(
          "__iter__",
          [](onnx2::utils::RepeatedField<onnx2::utils::String> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Iterates over the elements.");
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

  py::class_<onnx2::utils::String>(m, "String",
                                   "Simplified string with no final null character.")
      .def(py::init<std::string>())
      .def(
          "__str__",
          [](const onnx2::utils::String &self) -> std::string { return self.as_string(); },
          "Converts this instance into a python string.")
      .def(
          "__repr__",
          [](const onnx2::utils::String &self) -> std::string {
            return std::string("'") + self.as_string() + std::string("'");
          },
          "Represention with surrounding quotes.")
      .def(
          "__len__", [](const onnx2::utils::String &self) -> int { return self.size(); },
          "Returns the length of the string.")
      .def(
          "__eq__",
          [](const onnx2::utils::String &self, const std::string &s) -> int {
            return self == s;
          },
          "Compares two strings.");

  define_repeated_field_type<int64_t>(m, "RepeatedFieldInt64");
  define_repeated_field_type<int32_t>(m, "RepeatedFieldInt32");
  define_repeated_field_type<uint64_t>(m, "RepeatedFieldUInt64");
  define_repeated_field_type<float>(m, "RepeatedFieldFloat");
  define_repeated_field_type<double>(m, "RepeatedFieldDouble");
  define_repeated_field_type<onnx2::utils::String>(m, "RepeatedFieldString");

  py::enum_<onnx2::OperatorStatus>(m, "OperatorStatus", py::arithmetic())
      .value("EXPERIMENTAL", onnx2::OperatorStatus::EXPERIMENTAL)
      .value("STABLE", onnx2::OperatorStatus::STABLE)
      .export_values();

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

  py::class_<onnx2::Message>(m, "Message", "Message, base class for all onnx2 classes")
      .def(py::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value)
      .PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  define_repeated_field_type<onnx2::StringStringEntryProto>(
      m, "RepeatedFieldStringStringEntryProto");

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version)
      .PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  define_repeated_field_type<onnx2::OperatorSetIdProto>(m, "RepeatedFieldOperatorSetIdProto");

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names)
      .PYADD_PROTO_SERIALIZATION(TensorAnnotation);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value)
      .PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  define_repeated_field_type<onnx2::IntIntListEntryProto>(m,
                                                          "RepeatedFieldIntIntListEntryProto");

  PYDEFINE_PROTO(m, DeviceConfigurationProto)
      .PYFIELD_STR(DeviceConfigurationProto, name)
      .PYFIELD(DeviceConfigurationProto, num_devices)
      .PYFIELD(DeviceConfigurationProto, device)
      .PYADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  PYDEFINE_PROTO(m, SimpleShardedDimProto)
      .PYFIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .PYFIELD_STR(SimpleShardedDimProto, dim_param)
      .PYFIELD(SimpleShardedDimProto, num_shards)
      .PYADD_PROTO_SERIALIZATION(SimpleShardedDimProto);
  define_repeated_field_type<onnx2::SimpleShardedDimProto>(
      m, "RepeatedFieldSimpleShardedDimProto");

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding)
      .PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  define_repeated_field_type<onnx2::ShardedDimProto>(m, "RepeatedFieldShardedDimProto");

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim)
      .PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  define_repeated_field_type<onnx2::ShardingSpecProto>(m, "RepeatedFieldShardingSpecProto");

  PYDEFINE_PROTO(m, NodeDeviceConfigurationProto)
      .PYFIELD_STR(NodeDeviceConfigurationProto, configuration_id)
      .PYFIELD(NodeDeviceConfigurationProto, sharding_spec)
      .PYFIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage)
      .PYADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorShapeProto, cls_tensor_shape_proto);
  PYDEFINE_SUBPROTO(cls_tensor_shape_proto, TensorShapeProto, Dimension)
      .PYFIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .PYFIELD_STR(TensorShapeProto::Dimension, dim_param)
      .PYFIELD_STR(TensorShapeProto::Dimension, denotation)
      .PYADD_PROTO_SERIALIZATION(TensorShapeProto::Dimension);
  define_repeated_field_type<onnx2::TensorShapeProto::Dimension>(m, "RepeatedFieldDimension");
  cls_tensor_shape_proto.PYFIELD(TensorShapeProto, dim)
      .PYADD_PROTO_SERIALIZATION(TensorShapeProto);

  PYDEFINE_PROTO(m, TensorProto)
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
      .PYFIELD(TensorProto, dims)
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
          onnx2::TensorProto::DOC_data_type)
      .PYFIELD_STR(TensorProto, name)
      .PYFIELD_STR(TensorProto, doc_string)
      .PYFIELD(TensorProto, external_data)
      .PYFIELD(TensorProto, metadata_props)
      .PYFIELD(TensorProto, dims)
      .PYFIELD(TensorProto, double_data)
      .PYFIELD(TensorProto, float_data)
      .PYFIELD(TensorProto, int64_data)
      .PYFIELD(TensorProto, int32_data)
      .PYFIELD(TensorProto, uint64_data)
      .def_property(
          "string_data",
          [](const onnx2::TensorProto &self) -> py::list {
            py::list result;
            for (const auto &s : self.string_data_) {
              result.append(py::bytes(std::string(s.data(), s.size())));
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
          onnx2::TensorProto::DOC_string_data)
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
          onnx2::TensorProto::DOC_raw_data)
      .PYADD_PROTO_SERIALIZATION(TensorProto);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims)
      .PYADD_PROTO_SERIALIZATION(SparseTensorProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TypeProto, cls_type_proto);
  PYDEFINE_SUBPROTO(cls_type_proto, TypeProto, Tensor)
      .PYFIELD_OPTIONAL_INT(TypeProto::Tensor, elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape)
      .PYADD_PROTO_SERIALIZATION(TypeProto::Tensor);
  PYDEFINE_SUBPROTO(cls_type_proto, TypeProto, SparseTensor)
      .PYFIELD_OPTIONAL_INT(TypeProto::SparseTensor, elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape)
      .PYADD_PROTO_SERIALIZATION(TypeProto::SparseTensor);
  PYDEFINE_SUBPROTO(cls_type_proto, TypeProto, Sequence)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type)
      .PYADD_PROTO_SERIALIZATION(TypeProto::Sequence);
  PYDEFINE_SUBPROTO(cls_type_proto, TypeProto, Optional)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type)
      .PYADD_PROTO_SERIALIZATION(TypeProto::Optional);
  PYDEFINE_SUBPROTO(cls_type_proto, TypeProto, Map)
      .PYFIELD(TypeProto::Map, key_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Map, value_type)
      .PYADD_PROTO_SERIALIZATION(TypeProto::Map);
  cls_type_proto.PYFIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .PYFIELD_STR(TypeProto, denotation)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, optional_type)
      .PYADD_PROTO_SERIALIZATION(TypeProto);
}
