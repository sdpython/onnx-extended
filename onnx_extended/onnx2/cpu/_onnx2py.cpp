#include "onnx2.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define PYDEFINE_PROTO(m, cls)                                                                         \
  py::class_<onnx2::cls, onnx2::Message>(m, #cls, onnx2::cls::DOC).def(py::init<>())

#define PYDEFINE_SUBPROTO(m, cls, subname)                                                             \
  py::class_<onnx2::cls::subname, onnx2::Message>(m, #subname, onnx2::cls::subname::DOC)               \
      .def(py::init<>())

#define PYDEFINE_PROTO_WITH_SUBTYPES(m, cls, name)                                                     \
  py::class_<onnx2::cls, onnx2::Message> name(m, #cls, onnx2::cls::DOC);                               \
  name.def(py::init<>());

#define PYADD_PROTO_SERIALIZATION(cls)                                                                 \
  def(                                                                                                 \
      "ParseFromString",                                                                               \
      [](onnx2::cls &self, py::bytes data) {                                                           \
        std::string raw = data;                                                                        \
        self.ParseFromString(raw);                                                                     \
      },                                                                                               \
      "Parses a sequence of bytes to fill this instance.")                                             \
      .def(                                                                                            \
          "SerializeToString",                                                                         \
          [](onnx2::cls &self) {                                                                       \
            std::string out;                                                                           \
            self.SerializeToString(out);                                                               \
            return py::bytes(out);                                                                     \
          },                                                                                           \
          "Serializes this instance into a sequence of bytes.")                                        \
      .def(                                                                                            \
          "__str__",                                                                                   \
          [](onnx2::cls &self) -> std::string {                                                        \
            std::vector<std::string> rows = self.SerializeToVectorString();                            \
            return onnx2::utils::join_string(rows);                                                    \
          },                                                                                           \
          "Creates a printable string for this class.")                                                \
      .def(                                                                                            \
          "CopyFrom", [](onnx2::cls &self, const onnx2::cls &src) { self.CopyFrom(src); },             \
          "Copy one instance into this one.")                                                          \
      .def(                                                                                            \
          "__eq__",                                                                                    \
          [](const onnx2::cls &self, const onnx2::cls &src) -> bool {                                  \
            std::string s1;                                                                            \
            self.SerializeToString(s1);                                                                \
            std::string s2;                                                                            \
            src.SerializeToString(s2);                                                                 \
            return s1 == s2;                                                                           \
          },                                                                                           \
          "Compares the serialized strings.")

#define PYFIELD(cls, name)                                                                             \
  def_readwrite(#name, &onnx2::cls::name##_, #name)                                                    \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_STR(cls, name)                                                                         \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](const onnx2::cls &self) -> std::string {                                                      \
        std::string s = self.ref_##name().as_string();                                                 \
        return s;                                                                                      \
      },                                                                                               \
      [](onnx2::cls &self, py::object obj) {                                                           \
        if (py::isinstance<py::str>(obj)) {                                                            \
          std::string st = obj.cast<std::string>();                                                    \
          self.set_##name(st);                                                                         \
        } else if (py::isinstance<py::bytes>(obj)) {                                                   \
          std::string st = obj.cast<py::bytes>();                                                      \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(obj.cast<onnx2::cls::name##_t &>());                                         \
        }                                                                                              \
      },                                                                                               \
      onnx2::cls::DOC_##name)                                                                          \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value")

#define PYFIELD_STR_AS_BYTES(cls, name)                                                                \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](const onnx2::cls &self) -> py::bytes {                                                        \
        std::string s = py::bytes(self.ref_##name().as_string());                                      \
        return s;                                                                                      \
      },                                                                                               \
      [](onnx2::cls &self, py::object obj) {                                                           \
        if (py::isinstance<py::str>(obj)) {                                                            \
          std::string st = obj.cast<std::string>();                                                    \
          self.set_##name(st);                                                                         \
        } else if (py::isinstance<py::bytes>(obj)) {                                                   \
          std::string st = obj.cast<py::bytes>();                                                      \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(obj.cast<onnx2::cls::name##_t &>());                                         \
        }                                                                                              \
      },                                                                                               \
      onnx2::cls::DOC_##name)                                                                          \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value")

#define _PYFIELD_OPTIONAL_CTYPE(cls, name, ctype)                                                      \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](onnx2::cls &self) -> py::object {                                                             \
        if (!self.has_##name())                                                                        \
          return py::none();                                                                           \
        return py::cast(self.ref_##name(), py::return_value_policy::reference);                        \
      },                                                                                               \
      [](onnx2::cls &self, py::object obj) {                                                           \
        if (obj.is_none()) {                                                                           \
          self.reset_##name();                                                                         \
        } else if (py::isinstance<py::ctype##_>(obj)) {                                                \
          self.set_##name(obj.cast<ctype>());                                                          \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      onnx2::cls::DOC_##name)                                                                          \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_INT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, int)
#define PYFIELD_OPTIONAL_FLOAT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, float)

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                              \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](onnx2::cls &self) -> py::object {                                                             \
        if (!self.name##_.has_value()) {                                                               \
          if (self.has_oneof_##name())                                                                 \
            return py::none();                                                                         \
          self.name##_.set_empty_value();                                                              \
        }                                                                                              \
        return py::cast(self.name##_.value, py::return_value_policy::reference);                       \
      },                                                                                               \
      [](onnx2::cls &self, py::object obj) {                                                           \
        if (obj.is_none()) {                                                                           \
          self.name##_.reset();                                                                        \
        } else if (py::isinstance<onnx2::cls::name##_t>(obj)) {                                        \
          self.name##_ = obj.cast<onnx2::cls::name##_t &>();                                           \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      onnx2::cls::DOC_##name)                                                                          \
      .def("has_" #name, &onnx2::cls::has_##name, "Tells if '" #name "' has a value.")                 \
      .def(                                                                                            \
          "add_" #name, [](onnx2::cls & self) -> onnx2::cls::name##_t & {                              \
            self.name##_.set_empty_value();                                                            \
            return *self.name##_;                                                                      \
          },                                                                                           \
          py::return_value_policy::reference, "Sets an empty value.")

#define SHORTEN_CODE(cls, dtype)                                                                       \
  def_property_readonly_static(#dtype,                                                                 \
                               [](py::object) -> int { return static_cast<int>(onnx2::cls::dtype); })

#define DECLARE_REPEATED_FIELD(T, inst_name)                                                           \
  py::class_<onnx2::utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);
#define DECLARE_REPEATED_FIELD_PROTO(T, inst_name)                                                     \
  py::class_<onnx2::utils::RepeatedField<onnx2::T>> inst_name(m, "RepeatedField" #T,                   \
                                                              "RepeatedField" #T);
#define DECLARE_REPEATED_FIELD_SUBPROTO(cls, T, inst_name)                                             \
  py::class_<onnx2::utils::RepeatedField<onnx2::cls::T>> inst_name(m, "RepeatedField" #cls #T,         \
                                                                   "RepeatedField" #cls #T);

template <typename T>
void define_repeated_field_type(py::class_<onnx2::utils::RepeatedField<T>> &pycls) {
  pycls.def(py::init<>())
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
          "__iter__",
          [](onnx2::utils::RepeatedField<T> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Iterates over the elements.");
}

template <typename T>
void define_repeated_field_type_extend(py::class_<onnx2::utils::RepeatedField<T>> &pycls) {
  pycls
      .def(
          "append", [](onnx2::utils::RepeatedField<T> &self, T v) { self.push_back(v); },
          py::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](onnx2::utils::RepeatedField<T> &self, py::iterable iterable) {
            if (py::isinstance<onnx2::utils::RepeatedField<T>>(iterable)) {
              self.extend(iterable.cast<onnx2::utils::RepeatedField<T> &>());
            } else {
              self.extend(iterable.cast<std::vector<T>>());
            }
          },
          py::arg("sequence"), "Extends the list of values.");
}

template <>
void define_repeated_field_type_extend(
    py::class_<onnx2::utils::RepeatedField<onnx2::utils::String>> &pycls) {
  pycls
      .def(
          "append",
          [](onnx2::utils::RepeatedField<onnx2::utils::String> &self, const onnx2::utils::String &v) {
            self.push_back(v);
          },
          py::arg("item"), "Append one element to the list of values.")
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
          py::arg("sequence"), "Extends the list of values.");
}

template <typename T>
void define_repeated_field_type_extend_list(py::class_<onnx2::utils::RepeatedField<T>> &pycls) {
  pycls
      .def(
          "append", [](onnx2::utils::RepeatedField<T> &self, const T &v) { self.push_back(v); },
          py::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](onnx2::utils::RepeatedField<T> &self, py::iterable iterable) {
            if (py::isinstance<onnx2::utils::RepeatedField<T>>(iterable)) {
              self.extend(iterable.cast<onnx2::utils::RepeatedField<T> &>());
            } else {
              py::list els = iterable.cast<py::list>();
              for (auto it : els) {
                if (py::isinstance<const T &>(it)) {
                  self.push_back(it.cast<T>());
                } else if (py::isinstance<T>(it)) {
                  self.push_back(it.cast<T>());
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          py::arg("sequence"), "Extends the list of values.");
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

  py::class_<onnx2::utils::String>(m, "String", "Simplified string with no final null character.")
      .def(py::init<std::string>())
      .def(
          "__str__", [](const onnx2::utils::String &self) -> std::string { return self.as_string(); },
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
          [](const onnx2::utils::String &self, const std::string &s) -> int { return self == s; },
          "Compares two strings.");

  DECLARE_REPEATED_FIELD(int64_t, rep_int64_t);
  define_repeated_field_type(rep_int64_t);
  define_repeated_field_type_extend(rep_int64_t);

  DECLARE_REPEATED_FIELD(int32_t, rep_int32_t);
  define_repeated_field_type(rep_int32_t);
  define_repeated_field_type_extend(rep_int32_t);

  DECLARE_REPEATED_FIELD(uint64_t, rep_uint64_t);
  define_repeated_field_type(rep_uint64_t);
  define_repeated_field_type_extend(rep_uint64_t);

  DECLARE_REPEATED_FIELD(float, rep_float);
  define_repeated_field_type(rep_float);
  define_repeated_field_type_extend(rep_float);

  DECLARE_REPEATED_FIELD(double, rep_double);
  define_repeated_field_type(rep_double);
  define_repeated_field_type_extend(rep_double);

  py::class_<onnx2::utils::RepeatedField<onnx2::utils::String>> rep_string(m, "RepeatedFieldString",
                                                                           "RepeatedFieldString");
  define_repeated_field_type(rep_string);
  define_repeated_field_type_extend(rep_string);

  py::enum_<onnx2::OperatorStatus>(m, "OperatorStatus", py::arithmetic())
      .value("EXPERIMENTAL", onnx2::OperatorStatus::EXPERIMENTAL)
      .value("STABLE", onnx2::OperatorStatus::STABLE)
      .export_values();

  py::class_<onnx2::Message>(m, "Message", "Message, base class for all onnx2 classes")
      .def(py::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value)
      .PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(StringStringEntryProto, rep_ssentry);
  define_repeated_field_type(rep_ssentry);
  define_repeated_field_type_extend_list(rep_ssentry);

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version)
      .PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  DECLARE_REPEATED_FIELD_PROTO(OperatorSetIdProto, rep_osp);
  define_repeated_field_type(rep_osp);
  define_repeated_field_type_extend_list(rep_osp);

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names)
      .PYADD_PROTO_SERIALIZATION(TensorAnnotation);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value)
      .PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(IntIntListEntryProto, rep_iil);
  define_repeated_field_type(rep_iil);
  define_repeated_field_type_extend_list(rep_iil);

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
  DECLARE_REPEATED_FIELD_PROTO(SimpleShardedDimProto, rep_ssdp);
  define_repeated_field_type(rep_ssdp);
  define_repeated_field_type_extend_list(rep_ssdp);

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding)
      .PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardedDimProto, rep_sdp);
  define_repeated_field_type(rep_sdp);
  define_repeated_field_type_extend_list(rep_sdp);

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim)
      .PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardingSpecProto, rep_ssp);
  define_repeated_field_type(rep_ssp);
  define_repeated_field_type_extend_list(rep_ssp);

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
  DECLARE_REPEATED_FIELD_SUBPROTO(TensorShapeProto, Dimension, rep_tspd);
  define_repeated_field_type(rep_tspd);
  define_repeated_field_type_extend_list(rep_tspd);
  cls_tensor_shape_proto.PYFIELD(TensorShapeProto, dim).PYADD_PROTO_SERIALIZATION(TensorShapeProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorProto, cls_tensor_proto);

  py::enum_<onnx2::TensorProto::DataType>(cls_tensor_proto, "DataType", py::arithmetic())
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
  cls_tensor_proto.SHORTEN_CODE(TensorProto::DataType, UNDEFINED)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT)
      .SHORTEN_CODE(TensorProto::DataType, UINT8)
      .SHORTEN_CODE(TensorProto::DataType, INT8)
      .SHORTEN_CODE(TensorProto::DataType, UINT16)
      .SHORTEN_CODE(TensorProto::DataType, INT16)
      .SHORTEN_CODE(TensorProto::DataType, INT32)
      .SHORTEN_CODE(TensorProto::DataType, INT64)
      .SHORTEN_CODE(TensorProto::DataType, STRING)
      .SHORTEN_CODE(TensorProto::DataType, BOOL)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, DOUBLE)
      .SHORTEN_CODE(TensorProto::DataType, UINT32)
      .SHORTEN_CODE(TensorProto::DataType, UINT64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX128)
      .SHORTEN_CODE(TensorProto::DataType, BFLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FN)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, UINT4)
      .SHORTEN_CODE(TensorProto::DataType, INT4)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT4E2M1)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E8M0)
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
  DECLARE_REPEATED_FIELD_PROTO(TensorProto, rep_tp);
  define_repeated_field_type(rep_tp);
  define_repeated_field_type_extend_list(rep_tp);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims)
      .PYADD_PROTO_SERIALIZATION(SparseTensorProto);
  DECLARE_REPEATED_FIELD_PROTO(SparseTensorProto, rep_tsp);
  define_repeated_field_type(rep_tsp);
  define_repeated_field_type_extend_list(rep_tsp);

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

  PYDEFINE_PROTO(m, ValueInfoProto)
      .PYFIELD_STR(ValueInfoProto, name)
      .PYFIELD_OPTIONAL_PROTO(ValueInfoProto, type)
      .PYFIELD_STR(ValueInfoProto, doc_string)
      .PYFIELD(ValueInfoProto, metadata_props)
      .PYADD_PROTO_SERIALIZATION(ValueInfoProto);
  DECLARE_REPEATED_FIELD_PROTO(ValueInfoProto, rep_vip);
  define_repeated_field_type(rep_vip);
  define_repeated_field_type_extend_list(rep_vip);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, AttributeProto, cls_attribute_proto);
  py::enum_<onnx2::AttributeProto::AttributeType> attribute_type(cls_attribute_proto, "AttributeType",
                                                                 py::arithmetic());
  attribute_type.value("UNDEFINED", onnx2::AttributeProto::AttributeType::UNDEFINED)
      .value("FLOAT", onnx2::AttributeProto::AttributeType::FLOAT)
      .value("INT", onnx2::AttributeProto::AttributeType::INT)
      .value("STRING", onnx2::AttributeProto::AttributeType::STRING)
      .value("GRAPH", onnx2::AttributeProto::AttributeType::GRAPH)
      .value("SPARSE_TENSOR", onnx2::AttributeProto::AttributeType::SPARSE_TENSOR)
      .value("FLOATS", onnx2::AttributeProto::AttributeType::FLOATS)
      .value("INTS", onnx2::AttributeProto::AttributeType::INTS)
      .value("STRINGS", onnx2::AttributeProto::AttributeType::STRINGS)
      .value("GRAPHS", onnx2::AttributeProto::AttributeType::GRAPHS)
      .value("SPARSE_TENSORS", onnx2::AttributeProto::AttributeType::SPARSE_TENSORS)
      .export_values();
  attribute_type
      .def_static("items",
                  []() {
                    return std::vector<std::pair<std::string, onnx2::AttributeProto::AttributeType>>{
                        {"UNDEFINED", onnx2::AttributeProto::AttributeType::UNDEFINED},
                        {"FLOAT", onnx2::AttributeProto::AttributeType::FLOAT},
                        {"INT", onnx2::AttributeProto::AttributeType::INT},
                        {"STRING", onnx2::AttributeProto::AttributeType::STRING},
                        {"GRAPH", onnx2::AttributeProto::AttributeType::GRAPH},
                        {"SPARSE_TENSOR", onnx2::AttributeProto::AttributeType::SPARSE_TENSOR},
                        {"FLOATS", onnx2::AttributeProto::AttributeType::FLOATS},
                        {"INTS", onnx2::AttributeProto::AttributeType::INTS},
                        {"STRINGS", onnx2::AttributeProto::AttributeType::STRINGS},
                        {"GRAPHS", onnx2::AttributeProto::AttributeType::GRAPHS},
                        {"SPARSE_TENSORS", onnx2::AttributeProto::AttributeType::SPARSE_TENSORS},
                    };
                  })
      .def_static("keys",
                  []() {
                    return std::vector<std::string>{
                        "UNDEFINED", "FLOAT", "INT",     "STRING", "GRAPH",          "SPARSE_TENSOR",
                        "FLOATS",    "INTS",  "STRINGS", "GRAPHS", "SPARSE_TENSORS",
                    };
                  })
      .def_static("values", []() {
        return std::vector<onnx2::AttributeProto::AttributeType>{
            onnx2::AttributeProto::AttributeType::UNDEFINED,
            onnx2::AttributeProto::AttributeType::FLOAT,
            onnx2::AttributeProto::AttributeType::INT,
            onnx2::AttributeProto::AttributeType::STRING,
            onnx2::AttributeProto::AttributeType::GRAPH,
            onnx2::AttributeProto::AttributeType::SPARSE_TENSOR,
            onnx2::AttributeProto::AttributeType::FLOATS,
            onnx2::AttributeProto::AttributeType::INTS,
            onnx2::AttributeProto::AttributeType::STRINGS,
            onnx2::AttributeProto::AttributeType::GRAPHS,
            onnx2::AttributeProto::AttributeType::SPARSE_TENSORS,
        };
      });

  cls_attribute_proto.SHORTEN_CODE(AttributeProto::AttributeType, UNDEFINED)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOAT)
      .SHORTEN_CODE(AttributeProto::AttributeType, INT)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRING)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPH)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOATS)
      .SHORTEN_CODE(AttributeProto::AttributeType, INTS)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRINGS)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSORS)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPHS)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSORS)
      .PYFIELD_STR(AttributeProto, name)
      .PYFIELD_STR(AttributeProto, ref_attr_name)
      .PYFIELD_STR(AttributeProto, doc_string)
      .def_property(
          "type",
          [](const onnx2::AttributeProto &self) -> onnx2::AttributeProto::AttributeType {
            return self.type_;
          },
          [](onnx2::AttributeProto &self, py::object obj) {
            if (py::isinstance<py::int_>(obj)) {
              self.type_ = static_cast<onnx2::AttributeProto::AttributeType>(obj.cast<int>());
            } else {
              self.type_ = obj.cast<onnx2::AttributeProto::AttributeType>();
            }
          },
          onnx2::AttributeProto::DOC_type)
      .PYFIELD_OPTIONAL_FLOAT(AttributeProto, f)
      .PYFIELD_OPTIONAL_INT(AttributeProto, i)
      .PYFIELD_STR_AS_BYTES(AttributeProto, s)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, t)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, sparse_tensor)
      //.PYFIELD_OPTIONAL_PROTO(AttributeProto, g)
      .PYFIELD(AttributeProto, floats)
      .PYFIELD(AttributeProto, ints)
      .PYFIELD(AttributeProto, strings)
      .PYFIELD(AttributeProto, tensors)
      .PYFIELD(AttributeProto, sparse_tensors)
      //.PYFIELD(AttributeProto, graphs)
      .PYADD_PROTO_SERIALIZATION(AttributeProto);
  DECLARE_REPEATED_FIELD_PROTO(AttributeProto, rep_ap);
  define_repeated_field_type(rep_ap);
  define_repeated_field_type_extend_list(rep_ap);
}
