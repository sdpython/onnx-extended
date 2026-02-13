#include "onnx2.h"
#include "onnx2_helper.h"
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace onnx2;

#define PYDEFINE_PROTO(m, cls)                                                                         \
  nb::class_<cls, Message> nb_##cls(m, #cls, cls::DOC);                                                \
  nb_##cls.def(nb::init<>())

#define PYDEFINE_SUBPROTO(m, cls, subname)                                                             \
  nb::class_<cls::subname, Message> nb_sub_##cls##subname(m, #subname, cls::subname::DOC);             \
  nb_sub_##cls##subname.def(nb::init<>())

#define PYDEFINE_PROTO_WITH_SUBTYPES(m, cls)                                                           \
  nb::class_<cls, Message> nb_##cls(m, #cls, cls::DOC);                                                \
  nb_##cls.def(nb::init<>());

#define PYDEFINE_PROTO_WITH_SUBTYPES2(m, cls, subcls)                                                  \
  nb::class_<cls::subcls, Message> nb_sub_##cls##subcls(nb_##cls, #subcls, cls::subcls::DOC);          \
  nb_sub_##cls##subcls.def(nb::init<>());

#define _PYADD_PROTO_SERIALIZATION(cls, name_inst) pyadd_proto_serialization(name_inst);

#define PYADD_PROTO_SERIALIZATION(cls) _PYADD_PROTO_SERIALIZATION(cls, nb_##cls)
#define PYADD_SUBPROTO_SERIALIZATION(cls, sub) _PYADD_PROTO_SERIALIZATION(cls::sub, nb_sub_##cls##sub)

#define PYFIELD(cls, name)                                                                             \
  def_rw(#name, &cls::name##_, cls::DOC_##name)                                                        \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_STR(cls, name)                                                                         \
  def_prop_rw(                                                                                         \
      #name,                                                                                           \
      [](const cls &self) -> std::string {                                                             \
        std::string s = self.ref_##name().as_string();                                                 \
        return s;                                                                                      \
      },                                                                                               \
      [](cls &self, nb::object obj) {                                                                  \
        if (nb::isinstance<nb::str>(obj)) {                                                            \
          std::string st = nb::cast<std::string>(obj);                                                 \
          self.set_##name(st);                                                                         \
        } else if (nb::isinstance<nb::bytes>(obj)) {                                                   \
          nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(obj);                                      \
          std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());               \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(nb::cast<cls::name##_t &>(obj));                                             \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define PYFIELD_STR_AS_BYTES(cls, name)                                                                \
  def_prop_rw(                                                                                         \
      #name,                                                                                           \
      [](const cls &self) -> nb::bytes {                                                               \
        std::string s = self.ref_##name().as_string();                                                 \
        return nb::bytes(s.data(), s.size());                                                          \
      },                                                                                               \
      [](cls &self, nb::object obj) {                                                                  \
        if (nb::isinstance<nb::str>(obj)) {                                                            \
          std::string st = nb::cast<std::string>(obj);                                                 \
          self.set_##name(st);                                                                         \
        } else if (nb::isinstance<nb::bytes>(obj)) {                                                   \
          nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(obj);                                      \
          std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());               \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(nb::cast<cls::name##_t &>(obj));                                             \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define _PYFIELD_OPTIONAL_CTYPE(cls, name, ctype)                                                      \
  def_prop_rw(                                                                                         \
      #name,                                                                                           \
      [](cls &self) -> nb::object {                                                                    \
        if (!self.has_##name())                                                                        \
          return nb::none();                                                                           \
        return nb::cast(self.ref_##name(), nb::rv_policy::reference);                                  \
      },                                                                                               \
      [](cls &self, nb::object obj) {                                                                  \
        if (obj.is_none()) {                                                                           \
          self.reset_##name();                                                                         \
        } else if (nb::isinstance<nb::ctype##_>(obj)) {                                                \
          self.set_##name(nb::cast<ctype>(obj));                                                       \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_INT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, int)
#define PYFIELD_OPTIONAL_FLOAT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, float)

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                              \
  def_prop_rw(                                                                                         \
      #name,                                                                                           \
      [](cls &self) -> nb::object {                                                                    \
        if (!self.name##_.has_value()) {                                                               \
          if (self.has_oneof_##name())                                                                 \
            return nb::none();                                                                         \
          self.name##_.set_empty_value();                                                              \
        }                                                                                              \
        return nb::cast(*self.name##_, nb::rv_policy::reference);                                      \
      },                                                                                               \
      [](cls &self, nb::object obj) {                                                                  \
        if (obj.is_none()) {                                                                           \
          self.name##_.reset();                                                                        \
        } else if (nb::isinstance<cls::name##_t>(obj)) {                                               \
          self.name##_ = nb::cast<cls::name##_t &>(obj);                                               \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")                        \
      .def(                                                                                            \
          "add_" #name, [](cls & self)->cls::name##_t & {                                              \
            self.name##_.set_empty_value();                                                            \
            return *self.name##_;                                                                      \
          },                                                                                           \
          nb::rv_policy::reference, "Sets an empty value.")

#define SHORTEN_CODE(cls, dtype)                                                                       \
  def_prop_ro_static(#dtype, [](nb::handle) -> int { return static_cast<int>(cls::dtype); })

#define DECLARE_REPEATED_FIELD(T, inst_name)                                                           \
  nb::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);

#define DECLARE_REPEATED_FIELD_PROTO(T, inst_name)                                                     \
  nb::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);            \
  nb::class_<utils::RepeatedProtoField<T>> inst_name##_proto(m, "RepeatedProtoField" #T,               \
                                                             "RepeatedProtoField" #T);

#define DECLARE_REPEATED_FIELD_SUBPROTO(cls, T, inst_name)                                             \
  nb::class_<utils::RepeatedField<cls::T>> inst_name(m, "RepeatedField" #cls #T,                       \
                                                     "RepeatedField" #cls #T);                         \
  nb::class_<utils::RepeatedProtoField<cls::T>> inst_name##_proto(m, "RepeatedProtoField" #cls #T,     \
                                                                  "RepeatedProtoField" #cls #T);

template <typename cls> void pyadd_proto_serialization(nb::class_<cls, Message> &name_inst) {
  name_inst
      .def(
          "ParseFromString",
          [](cls &self, nb::bytes data, nb::object options) {
            std::string raw(static_cast<const char *>(data.data()), data.size());
            if (nb::isinstance<ParseOptions &>(options)) {
              self.ParseFromString(raw, nb::cast<ParseOptions &>(options));
            } else {
              self.ParseFromString(raw);
            }
          },
          nb::arg("data"), nb::arg("options") = nb::none(),
          "Parses a sequence of bytes to fill this instance.")
      .def(
          "ParseFromString",
          [](cls &self, const std::string &raw, nb::object options) {
            if (nb::isinstance<ParseOptions &>(options)) {
              self.ParseFromString(raw, nb::cast<ParseOptions &>(options));
            } else {
              self.ParseFromString(raw);
            }
          },
          nb::arg("data"), nb::arg("options") = nb::none(), "Parses a string to fill this instance.")
      .def(
          "ParseFromFile",
          [](cls &self, const std::string &file_path, nb::object options,
             const std::string &external_data_file) {
            utils::FileStream *stream = external_data_file.empty()
                                            ? new utils::FileStream(file_path)
                                            : new utils::TwoFilesStream(file_path, external_data_file);
            if (nb::isinstance<ParseOptions &>(options)) {
              ParseOptions &coptions = nb::cast<ParseOptions &>(options);
              if (coptions.parallel) {
                stream->StartThreadPool(coptions.num_threads);
              }
              ParseProtoFromStream(self, *stream, coptions);
              if (coptions.parallel) {
                stream->WaitForDelayedBlock();
              }
            } else {
              ParseOptions opts;
              ParseProtoFromStream(self, *stream, opts);
            }
            delete stream;
          },
          nb::arg("name"), nb::arg("options") = nb::none(), nb::arg("external_data_file") = "",
          "Parses a binary file to fill this instance.")
      .def(
          "SerializeSize",
          [](cls &self, nb::object options) -> uint64_t {
            if (nb::isinstance<SerializeOptions &>(options)) {
              utils::StringWriteStream out;
              return self.SerializeSize(out, nb::cast<SerializeOptions &>(options));
            } else {
              return self.SerializeSize();
            }
          },
          nb::arg("options") = nb::none(), "Returns the size once serialized without serializing.")
      .def(
          "SerializeToString",
          [](cls &self, nb::object options) {
            std::string out;
            if (nb::isinstance<SerializeOptions &>(options)) {
              self.SerializeToString(out, nb::cast<SerializeOptions &>(options));
            } else {
              SerializeOptions opts;
              self.SerializeToString(out, opts);
            }
            return nb::bytes(out.data(), out.size());
          },
          nb::arg("options") = nb::none(), "Serializes this instance into a sequence of bytes.")
      .def(
          "SerializeToFile",
          [](cls &self, const std::string &file_path, nb::object options,
             std::string &external_data_file) {
            utils::BinaryWriteStream *stream =
                external_data_file.empty()
                    ? new utils::FileWriteStream(file_path)
                    : new utils::TwoFilesWriteStream(file_path, external_data_file);
            if (nb::isinstance<SerializeOptions &>(options)) {
              SerializeProtoToStream(self, *stream, nb::cast<SerializeOptions &>(options),
                                     !external_data_file.empty());
            } else {
              SerializeOptions opts;
              SerializeProtoToStream(self, *stream, opts, !external_data_file.empty());
            }
            delete stream;
          },
          nb::arg("name"), nb::arg("options") = nb::none(), nb::arg("external_data_file") = "",
          "Serializes this instance into a file. If ``external_data_size`` is not empty, big weights "
          "are stored in this (depending on ``options.raw_data_threshold``.")
      .def(
          "__str__",
          [](cls &self) -> std::string {
            utils::PrintOptions opts;
            std::vector<std::string> rows = self.PrintToVectorString(opts);
            return utils::join_string(rows);
          },
          "Creates a printable string for this class.")
      .def(
          "CopyFrom", [](cls &self, const cls &src) { self.CopyFrom(src); },
          "Copies one instance into this one.")
      .def(
          "__eq__",
          [](const cls &self, const cls &other) -> bool {
            SerializeOptions opts1, opts2;
            std::string s1;
            self.SerializeToString(s1, opts1);
            std::string s2;
            other.SerializeToString(s2, opts2);
            return s1 == s2;
          },
          nb::arg("other"), "Compares the serialized strings.");
}

template <typename T> void define_repeated_field_type(nb::class_<utils::RepeatedField<T>> &nbcls) {
  nbcls.def(nb::init<>())
      .def("add", &utils::RepeatedField<T>::add, nb::rv_policy::reference, "Adds an empty element.")
      .def("clear", &utils::RepeatedField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedField<T>::size, "Returns the number of elements.")
      .def(
          "__getitem__",
          [](utils::RepeatedField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self[index];
          },
          nb::rv_policy::reference, nb::arg("index"), "Returns the element at position index.")
      .def(
          "__delitem__",
          [](utils::RepeatedField<T> &self, nb::slice slice) {
            auto tup = slice.compute(self.size());
            auto [start, stop, step, slice_length] = tup;
            self.remove_range(start, stop, step);
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedField<T> &self) {
            return nb::make_iterator(nb::type<utils::RepeatedField<T>>(), "iterator", self.begin(),
                                     self.end());
          },
          nb::keep_alive<0, 1>(), "Iterates over the elements.");
}

template <typename T>
void define_repeated_field_type_extend(nb::class_<utils::RepeatedField<T>> &nbcls) {
  nbcls
      .def(
          "append", [](utils::RepeatedField<T> &self, T v) { self.push_back(v); }, nb::arg("item"),
          "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<T> &>(iterable));
            } else {
              self.extend(nb::cast<std::vector<T>>(iterable));
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

template <>
void define_repeated_field_type_extend(nb::class_<utils::RepeatedField<utils::String>> &nbcls) {
  nbcls
      .def(
          "append",
          [](utils::RepeatedField<utils::String> &self, const utils::String &v) { self.push_back(v); },
          nb::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<utils::String> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<utils::String>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<utils::String> &>(iterable));
            } else {
              std::vector<utils::String> values;
              for (auto it : iterable) {
                if (nb::isinstance<utils::String>(it)) {
                  values.push_back(nb::cast<utils::String &>(it));
                } else if (nb::isinstance<nb::bytes>(it)) {
                  nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(it);
                  std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
                  values.push_back(utils::String(st));
                } else {
                  values.emplace_back(utils::String(nb::cast<std::string>(it)));
                }
              }
              self.extend(values);
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

template <typename T>
void define_repeated_field_type_proto(nb::class_<utils::RepeatedField<T>> &nbcls,
                                      nb::class_<utils::RepeatedProtoField<T>> &nbcls_proto) {
  define_repeated_field_type(nbcls);
  nbcls
      .def(
          "append", [](utils::RepeatedField<T> &self, const T &v) { self.push_back(v); },
          nb::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if (nb::isinstance<const T &>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else if (nb::isinstance<T>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
  nbcls_proto.def(nb::init<>())
      .def("add", &utils::RepeatedProtoField<T>::add, nb::rv_policy::reference,
           "Adds an empty element.")
      .def("clear", &utils::RepeatedProtoField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedProtoField<T>::size, "Returns the number of elements.")
      .def(
          "__getitem__",
          [](utils::RepeatedProtoField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self[index];
          },
          nb::rv_policy::reference, nb::arg("index"), "Returns the element at position index.")
      .def(
          "__delitem__",
          [](utils::RepeatedProtoField<T> &self, nb::slice slice) {
            auto tup = slice.compute(self.size());
            auto [start, stop, step, slice_length] = tup;
            self.remove_range(start, stop, step);
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedProtoField<T> &self) {
            return nb::make_iterator(nb::type<utils::RepeatedProtoField<T>>(), "iterator", self.begin(),
                                     self.end());
          },
          nb::keep_alive<0, 1>(), "Iterates over the elements.")
      .def(
          "__eq__",
          [](utils::RepeatedField<T> &self, nb::list &obj) -> bool {
            if (self.size() != obj.size())
              return false;
            for (size_t i = 0; i < self.size(); ++i) {
              if (!nb::isinstance<T &>(obj[i]))
                return false;
              std::string s1, s2;
              self[i].SerializeToString(s1);
              nb::cast<T &>(obj[i]).SerializeToString(s2);
              if (s1 != s2)
                return false;
            }
            return true;
          },
          "Compares the container to a list of objects.")
      .def(
          "append", [](utils::RepeatedProtoField<T> &self, const T &v) { self.push_back(v); },
          nb::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedProtoField<T> &self, nb::iterable iterable) {
            if (nb::isinstance<utils::RepeatedProtoField<T>>(iterable)) {
              self.extend(nb::cast<utils::RepeatedProtoField<T> &>(iterable));
            } else {
              for (auto it : iterable) {
                if (nb::isinstance<const T &>(it)) {
                  self.push_back(nb::cast<const T &>(it));
                } else if (nb::isinstance<T>(it)) {
                  self.push_back(nb::cast<T>(it));
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          nb::arg("sequence"), "Extends the list of values.");
}

NB_MODULE(_onnx2py, m) {
  m.doc() = "onnx from python without protobuf but using the same format";

  m.def(
      "utils_onnx2_read_varint64",
      [](nb::bytes data) -> nb::tuple {
        std::string raw(static_cast<const char *>(data.data()), data.size());
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
        utils::StringStream st(ptr, raw.size());
        int64_t value = st.next_int64();
        return nb::make_tuple(value, st.tell());
      },
      nb::arg("data"),
      R"pbdoc(Reads a int64_t (protobuf format)
:param data: bytes
:return: 2-tuple, value and number of read bytes
)pbdoc");

  nb::class_<ParseOptions>(m, "ParseOptions", "Parsing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &ParseOptions::skip_raw_data,
              "if true, raw data will not be read but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw("raw_data_threshold", &ParseOptions::raw_data_threshold,
              "if skip_raw_data is true, raw data will be read only if it is larger than the threshold")
      .def_rw("parallel", &ParseOptions::parallel, "parallelizes the reading of the big blocks")
      .def_rw("num_threads", &ParseOptions::num_threads,
              "number of threads to run in parallel if parallel is true, -1 for as many threads "
              "as the number of cores");

  nb::class_<SerializeOptions>(m, "SerializeOptions", "Serializing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &SerializeOptions::skip_raw_data,
              "if true, raw data will not be written but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw(
          "raw_data_threshold", &SerializeOptions::raw_data_threshold,
          "if skip_raw_data is true, raw data will be written only if it is larger than the threshold");

  nb::class_<utils::PrintOptions>(m, "PrintOptions", "Printing options for proto classes")
      .def(nb::init<>())
      .def_rw("skip_raw_data", &utils::PrintOptions::skip_raw_data,
              "if true, raw data will not be printed but skipped, tensors are not valid in that "
              "case  but the model structure is still available")
      .def_rw(
          "raw_data_threshold", &utils::PrintOptions::raw_data_threshold,
          "if skip_raw_data is true, raw data will be printed only if it is larger than the threshold");

  nb::class_<utils::String>(m, "String", "Simplified string with no final null character.")
      .def(nb::init<std::string>())
      .def(
          "__str__", [](const utils::String &self) -> std::string { return self.as_string(); },
          "Converts this instance into a python string.")
      .def(
          "__repr__",
          [](const utils::String &self) -> std::string {
            return std::string("'") + self.as_string() + std::string("'");
          },
          "Represention with surrounding quotes.")
      .def(
          "__len__", [](const utils::String &self) -> int { return self.size(); },
          "Returns the length of the string.")
      .def(
          "__eq__", [](const utils::String &self, const std::string &s) -> int { return self == s; },
          "Compares two strings.")
      .def(
          "__eq__",
          [](const utils::String &self, const nb::bytes &bytes_obj) -> int {
            std::string st(static_cast<const char *>(bytes_obj.data()), bytes_obj.size());
            return self == st;
          },
          "Compares to a byte string.");

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

  nb::class_<utils::RepeatedField<utils::String>> rep_string(m, "RepeatedFieldString",
                                                             "RepeatedFieldString");
  define_repeated_field_type(rep_string);
  define_repeated_field_type_extend(rep_string);

  nb::enum_<OperatorStatus>(m, "OperatorStatus", nb::is_arithmetic())
      .value("EXPERIMENTAL", OperatorStatus::EXPERIMENTAL)
      .value("STABLE", OperatorStatus::STABLE)
      .export_values();

  nb::class_<Message>(m, "Message", "Message, base class for all onnx2 classes").def(nb::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value);
  PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(StringStringEntryProto, rep_ssentry);
  define_repeated_field_type_proto(rep_ssentry, rep_ssentry_proto);

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version);
  PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  DECLARE_REPEATED_FIELD_PROTO(OperatorSetIdProto, rep_osp);
  define_repeated_field_type_proto(rep_osp, rep_osp_proto);

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names);
  PYADD_PROTO_SERIALIZATION(TensorAnnotation);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value);
  PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(IntIntListEntryProto, rep_iil);
  define_repeated_field_type_proto(rep_iil, rep_iil_proto);

  PYDEFINE_PROTO(m, DeviceConfigurationProto)
      .PYFIELD_STR(DeviceConfigurationProto, name)
      .PYFIELD(DeviceConfigurationProto, num_devices)
      .PYFIELD(DeviceConfigurationProto, device);
  PYADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  PYDEFINE_PROTO(m, SimpleShardedDimProto)
      .PYFIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .PYFIELD_STR(SimpleShardedDimProto, dim_param)
      .PYFIELD(SimpleShardedDimProto, num_shards);
  PYADD_PROTO_SERIALIZATION(SimpleShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(SimpleShardedDimProto, rep_ssdp);
  define_repeated_field_type_proto(rep_ssdp, rep_ssdp_proto);

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding);
  PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardedDimProto, rep_sdp);
  define_repeated_field_type_proto(rep_sdp, rep_sdp_proto);

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim);
  PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardingSpecProto, rep_ssp);
  define_repeated_field_type_proto(rep_ssp, rep_ssp_proto);

  PYDEFINE_PROTO(m, NodeDeviceConfigurationProto)
      .PYFIELD_STR(NodeDeviceConfigurationProto, configuration_id)
      .PYFIELD(NodeDeviceConfigurationProto, sharding_spec)
      .PYFIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage);
  PYADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorShapeProto);
  PYDEFINE_SUBPROTO(nb_TensorShapeProto, TensorShapeProto, Dimension)
      .PYFIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .PYFIELD_STR(TensorShapeProto::Dimension, dim_param)
      .PYFIELD_STR(TensorShapeProto::Dimension, denotation);
  PYADD_SUBPROTO_SERIALIZATION(TensorShapeProto, Dimension);
  DECLARE_REPEATED_FIELD_SUBPROTO(TensorShapeProto, Dimension, rep_tspd);
  define_repeated_field_type_proto(rep_tspd, rep_tspd_proto);
  nb_TensorShapeProto.PYFIELD(TensorShapeProto, dim);
  PYADD_PROTO_SERIALIZATION(TensorShapeProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorProto);

  nb::enum_<TensorProto::DataType>(nb_TensorProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", TensorProto::DataType::UNDEFINED)
      .value("FLOAT", TensorProto::DataType::FLOAT)
      .value("UINT8", TensorProto::DataType::UINT8)
      .value("INT8", TensorProto::DataType::INT8)
      .value("UINT16", TensorProto::DataType::UINT16)
      .value("INT16", TensorProto::DataType::INT16)
      .value("INT32", TensorProto::DataType::INT32)
      .value("INT64", TensorProto::DataType::INT64)
      .value("STRING", TensorProto::DataType::STRING)
      .value("BOOL", TensorProto::DataType::BOOL)
      .value("FLOAT16", TensorProto::DataType::FLOAT16)
      .value("DOUBLE", TensorProto::DataType::DOUBLE)
      .value("UINT32", TensorProto::DataType::UINT32)
      .value("UINT64", TensorProto::DataType::UINT64)
      .value("COMPLEX64", TensorProto::DataType::COMPLEX64)
      .value("COMPLEX128", TensorProto::DataType::COMPLEX128)
      .value("BFLOAT16", TensorProto::DataType::BFLOAT16)
      .value("FLOAT8E4M3FN", TensorProto::DataType::FLOAT8E4M3FN)
      .value("FLOAT8E4M3FNUZ", TensorProto::DataType::FLOAT8E4M3FNUZ)
      .value("FLOAT8E5M2", TensorProto::DataType::FLOAT8E5M2)
      .value("FLOAT8E5M2FNUZ", TensorProto::DataType::FLOAT8E5M2FNUZ)
      .value("UINT4", TensorProto::DataType::UINT4)
      .value("INT4", TensorProto::DataType::INT4)
      .value("FLOAT4E2M1", TensorProto::DataType::FLOAT4E2M1)
      .value("FLOAT8E8M0", TensorProto::DataType::FLOAT8E8M0)
      .value("UINT2", TensorProto::DataType::UINT2)
      .value("INT2", TensorProto::DataType::INT2)
      .export_values();
  nb::enum_<TensorProto::DataLocation>(nb_TensorProto, "DataLocation", nb::is_arithmetic())
      .value("DEFAULT", TensorProto::DataLocation::DEFAULT)
      .value("EXTERNAL", TensorProto::DataLocation::EXTERNAL)
      .export_values();
  nb_TensorProto.SHORTEN_CODE(TensorProto::DataType, UNDEFINED)
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
      .SHORTEN_CODE(TensorProto::DataType, UINT2)
      .SHORTEN_CODE(TensorProto::DataType, INT2)
      .PYFIELD(TensorProto, dims)
      .def_prop_rw(
          "data_type", [](const TensorProto &self) -> TensorProto::DataType { return self.data_type_; },
          [](TensorProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.data_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.data_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TensorProto::DOC_data_type)
      .def_prop_rw(
          "data_location",
          [](const TensorProto &self) -> TensorProto::DataLocation {
            return self.has_data_location() ? *self.data_location_ : TensorProto::DataLocation::DEFAULT;
          },
          [](TensorProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.data_location_ = static_cast<TensorProto::DataLocation>(nb::cast<int>(obj));
            } else {
              self.data_location_ = nb::cast<TensorProto::DataLocation>(obj);
            }
          },
          TensorProto::DOC_data_location)
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
      .def_prop_rw(
          "string_data",
          [](const TensorProto &self) -> nb::list {
            nb::list result;
            for (const auto &s : self.string_data_) {
              result.append(nb::bytes(std::string(s.data(), s.size()).c_str(), s.size()));
            }
            return result;
          },
          [](TensorProto &self, nb::list data) {
            self.string_data_.reserve(data.size());

            for (const auto &item : data) {
              if (nb::isinstance<nb::bytes>(item)) {
                nanobind::bytes bytes_obj = nb::borrow<nb::bytes>(item);
                self.string_data_.emplace_back(
                    std::string(static_cast<const char *>(bytes_obj.data()), bytes_obj.size()));
              } else if (nb::isinstance<nb::str>(item)) {
                self.string_data_.emplace_back(nb::cast<std::string>(item));
              } else {
                EXT_THROW("unable to convert one item from the list into a string")
              }
            }
          },
          TensorProto::DOC_string_data)
      .def_prop_rw(
          "raw_data",
          [](const TensorProto &self) -> nb::bytes {
            return nb::bytes(reinterpret_cast<const char *>(self.raw_data_.data()),
                             self.raw_data_.size());
          },
          [](TensorProto &self, nb::bytes data) {
            std::string raw(static_cast<const char *>(data.data()), data.size());
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            self.raw_data_.resize(raw.size());
            memcpy(self.raw_data_.data(), ptr, raw.size());
          },
          TensorProto::DOC_raw_data);
  PYADD_PROTO_SERIALIZATION(TensorProto);
  DECLARE_REPEATED_FIELD_PROTO(TensorProto, rep_tp);
  define_repeated_field_type_proto(rep_tp, rep_tp_proto);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims);
  PYADD_PROTO_SERIALIZATION(SparseTensorProto);
  DECLARE_REPEATED_FIELD_PROTO(SparseTensorProto, rep_tsp);
  define_repeated_field_type_proto(rep_tsp, rep_tsp_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TypeProto);

  PYDEFINE_PROTO_WITH_SUBTYPES2(m, TypeProto, Tensor);
  nb_sub_TypeProtoTensor
      .def_prop_rw(
          "elem_type",
          [](const TypeProto::Tensor &self) -> TensorProto::DataType { return *self.elem_type_; },
          [](TypeProto::Tensor &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TypeProto::Tensor::DOC_elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Tensor);

  PYDEFINE_PROTO_WITH_SUBTYPES2(m, TypeProto, SparseTensor);
  nb_sub_TypeProtoSparseTensor
      .def_prop_rw(
          "elem_type",
          [](const TypeProto::SparseTensor &self) -> TensorProto::DataType { return *self.elem_type_; },
          [](TypeProto::SparseTensor &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          TypeProto::SparseTensor::DOC_elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);

  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Sequence)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Sequence);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Optional)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Optional);
  PYDEFINE_SUBPROTO(nb_TypeProto, TypeProto, Map)
      .PYFIELD(TypeProto::Map, key_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Map, value_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Map);
  nb_TypeProto.PYFIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .PYFIELD_STR(TypeProto, denotation)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, optional_type);
  PYADD_PROTO_SERIALIZATION(TypeProto);

  PYDEFINE_PROTO(m, ValueInfoProto)
      .PYFIELD_STR(ValueInfoProto, name)
      .PYFIELD_OPTIONAL_PROTO(ValueInfoProto, type)
      .PYFIELD_STR(ValueInfoProto, doc_string)
      .PYFIELD(ValueInfoProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(ValueInfoProto);
  DECLARE_REPEATED_FIELD_PROTO(ValueInfoProto, rep_vip);
  define_repeated_field_type_proto(rep_vip, rep_vip_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, AttributeProto);
  nb::enum_<AttributeProto::AttributeType> attribute_type(nb_AttributeProto, "AttributeType",
                                                          nb::is_arithmetic());
  attribute_type.value("UNDEFINED", AttributeProto::AttributeType::UNDEFINED)
      .value("FLOAT", AttributeProto::AttributeType::FLOAT)
      .value("INT", AttributeProto::AttributeType::INT)
      .value("STRING", AttributeProto::AttributeType::STRING)
      .value("TENSOR", AttributeProto::AttributeType::TENSOR)
      .value("GRAPH", AttributeProto::AttributeType::GRAPH)
      .value("SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR)
      .value("FLOATS", AttributeProto::AttributeType::FLOATS)
      .value("INTS", AttributeProto::AttributeType::INTS)
      .value("STRINGS", AttributeProto::AttributeType::STRINGS)
      .value("TENSORS", AttributeProto::AttributeType::TENSORS)
      .value("GRAPHS", AttributeProto::AttributeType::GRAPHS)
      .value("SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS)
      .export_values();
  attribute_type
      .def_static(
          "items",
          []() {
            return std::vector<std::pair<std::string, AttributeProto::AttributeType>>{
                {"UNDEFINED", AttributeProto::AttributeType::UNDEFINED},
                {"FLOAT", AttributeProto::AttributeType::FLOAT},
                {"INT", AttributeProto::AttributeType::INT},
                {"STRING", AttributeProto::AttributeType::STRING},
                {"TENSOR", AttributeProto::AttributeType::TENSOR},
                {"GRAPH", AttributeProto::AttributeType::GRAPH},
                {"SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR},
                {"FLOATS", AttributeProto::AttributeType::FLOATS},
                {"INTS", AttributeProto::AttributeType::INTS},
                {"STRINGS", AttributeProto::AttributeType::STRINGS},
                {"TENSORS", AttributeProto::AttributeType::TENSORS},
                {"GRAPHS", AttributeProto::AttributeType::GRAPHS},
                {"SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS},
            };
          },
          "Returns the list of (name, type).")
      .def_static(
          "keys",
          []() {
            return std::vector<std::string>{
                "UNDEFINED", "FLOAT", "INT",     "STRING",  "TENSOR", "GRAPH",          "SPARSE_TENSOR",
                "FLOATS",    "INTS",  "STRINGS", "TENSORS", "GRAPHS", "SPARSE_TENSORS",
            };
          },
          "Returns the list of names.")
      .def_static(
          "values",
          []() {
            return std::vector<AttributeProto::AttributeType>{
                AttributeProto::AttributeType::UNDEFINED,
                AttributeProto::AttributeType::FLOAT,
                AttributeProto::AttributeType::INT,
                AttributeProto::AttributeType::STRING,
                AttributeProto::AttributeType::TENSOR,
                AttributeProto::AttributeType::GRAPH,
                AttributeProto::AttributeType::SPARSE_TENSOR,
                AttributeProto::AttributeType::FLOATS,
                AttributeProto::AttributeType::INTS,
                AttributeProto::AttributeType::STRINGS,
                AttributeProto::AttributeType::TENSORS,
                AttributeProto::AttributeType::GRAPHS,
                AttributeProto::AttributeType::SPARSE_TENSORS,
            };
          },
          "Returns the list of types.");

  nb_AttributeProto.SHORTEN_CODE(AttributeProto::AttributeType, UNDEFINED)
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
      .def_prop_rw(
          "type",
          [](const AttributeProto &self) -> AttributeProto::AttributeType { return self.type_; },
          [](AttributeProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.type_ = static_cast<AttributeProto::AttributeType>(nb::cast<int>(obj));
            } else {
              self.type_ = nb::cast<AttributeProto::AttributeType>(obj);
            }
          },
          AttributeProto::DOC_type)
      .PYFIELD_OPTIONAL_FLOAT(AttributeProto, f)
      .PYFIELD_OPTIONAL_INT(AttributeProto, i)
      .PYFIELD_STR_AS_BYTES(AttributeProto, s)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, t)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, sparse_tensor)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, g)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, tp)
      .PYFIELD(AttributeProto, floats)
      .PYFIELD(AttributeProto, ints)
      .PYFIELD(AttributeProto, strings)
      .PYFIELD(AttributeProto, tensors)
      .PYFIELD(AttributeProto, sparse_tensors)
      .PYFIELD(AttributeProto, graphs);
  PYADD_PROTO_SERIALIZATION(AttributeProto);
  DECLARE_REPEATED_FIELD_PROTO(AttributeProto, rep_ap);
  define_repeated_field_type_proto(rep_ap, rep_ap_proto);

  PYDEFINE_PROTO(m, NodeProto)
      .PYFIELD(NodeProto, input)
      .PYFIELD(NodeProto, output)
      .PYFIELD_STR(NodeProto, name)
      .PYFIELD_STR(NodeProto, op_type)
      .PYFIELD_STR(NodeProto, domain)
      .PYFIELD_STR(NodeProto, overload)
      .PYFIELD(NodeProto, attribute)
      .PYFIELD_STR(NodeProto, doc_string)
      .PYFIELD(NodeProto, metadata_props)
      .PYFIELD(NodeProto, device_configurations);
  PYADD_PROTO_SERIALIZATION(NodeProto);
  DECLARE_REPEATED_FIELD_PROTO(NodeProto, rep_node);
  define_repeated_field_type_proto(rep_node, rep_node_proto);

  PYDEFINE_PROTO(m, GraphProto)
      .PYFIELD(GraphProto, node)
      .PYFIELD_STR(GraphProto, name)
      .PYFIELD(GraphProto, initializer)
      .PYFIELD(GraphProto, sparse_initializer)
      .PYFIELD_STR(GraphProto, doc_string)
      .PYFIELD(GraphProto, input)
      .PYFIELD(GraphProto, output)
      .PYFIELD(GraphProto, value_info)
      .PYFIELD(GraphProto, quantization_annotation)
      .PYFIELD(GraphProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(GraphProto);
  DECLARE_REPEATED_FIELD_PROTO(GraphProto, rep_graph);
  define_repeated_field_type_proto(rep_graph, rep_graph_proto);

  PYDEFINE_PROTO(m, FunctionProto)
      .PYFIELD_STR(FunctionProto, name)
      .PYFIELD(FunctionProto, input)
      .PYFIELD(FunctionProto, output)
      .PYFIELD(FunctionProto, attribute)
      .PYFIELD(FunctionProto, attribute_proto)
      .PYFIELD(FunctionProto, node)
      .PYFIELD_STR(FunctionProto, doc_string)
      .PYFIELD(FunctionProto, opset_import)
      .PYFIELD(FunctionProto, value_info)
      .PYFIELD(FunctionProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(FunctionProto);
  DECLARE_REPEATED_FIELD_PROTO(FunctionProto, rep_function);
  define_repeated_field_type_proto(rep_function, rep_function_proto);

  PYDEFINE_PROTO(m, ModelProto)
      .PYFIELD_STR(ModelProto, producer_name)
      .PYFIELD_STR(ModelProto, producer_version)
      .PYFIELD_STR(ModelProto, domain)
      .PYFIELD(ModelProto, model_version)
      .PYFIELD_STR(ModelProto, doc_string)
      .PYFIELD_OPTIONAL_PROTO(ModelProto, graph)
      .PYFIELD(ModelProto, opset_import)
      .PYFIELD_OPTIONAL_INT(ModelProto, ir_version)
      .PYFIELD(ModelProto, metadata_props)
      .PYFIELD(ModelProto, functions)
      .PYFIELD(ModelProto, configuration);
  PYADD_PROTO_SERIALIZATION(ModelProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, SequenceProto);
  nb::enum_<SequenceProto::DataType>(nb_SequenceProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", SequenceProto::DataType::UNDEFINED)
      .value("TENSOR", SequenceProto::DataType::TENSOR)
      .value("SPARSE_TENSOR", SequenceProto::DataType::SPARSE_TENSOR)
      .value("SEQUENCE", SequenceProto::DataType::SEQUENCE)
      .value("MAP", SequenceProto::DataType::MAP)
      .value("OPTIONAL", SequenceProto::DataType::OPTIONAL)
      .export_values();
  nb_SequenceProto.SHORTEN_CODE(SequenceProto::DataType, UNDEFINED)
      .SHORTEN_CODE(SequenceProto::DataType, TENSOR)
      .SHORTEN_CODE(SequenceProto::DataType, SPARSE_TENSOR)
      .SHORTEN_CODE(SequenceProto::DataType, SEQUENCE)
      .SHORTEN_CODE(SequenceProto::DataType, MAP)
      .SHORTEN_CODE(SequenceProto::DataType, OPTIONAL);
  nb_SequenceProto.PYFIELD_STR(SequenceProto, name)
      .def_prop_rw(
          "elem_type",
          [](const SequenceProto &self) -> SequenceProto::DataType { return self.elem_type_; },
          [](SequenceProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<SequenceProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<SequenceProto::DataType>(obj);
            }
          },
          SequenceProto::DOC_elem_type)
      .PYFIELD(SequenceProto, tensor_values)
      .PYFIELD(SequenceProto, sparse_tensor_values)
      .PYFIELD(SequenceProto, sequence_values)
      .PYFIELD(SequenceProto, map_values)
      .PYFIELD(SequenceProto, optional_values);
  PYADD_PROTO_SERIALIZATION(SequenceProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, MapProto);
  nb_MapProto.PYFIELD_STR(MapProto, name)
      .def_prop_rw(
          "key_type", [](const MapProto &self) -> TensorProto::DataType { return self.key_type_; },
          [](MapProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.key_type_ = static_cast<TensorProto::DataType>(nb::cast<int>(obj));
            } else {
              self.key_type_ = nb::cast<TensorProto::DataType>(obj);
            }
          },
          MapProto::DOC_key_type)
      .PYFIELD(MapProto, keys)
      .PYFIELD(MapProto, string_keys)
      .PYFIELD(MapProto, values);
  PYADD_PROTO_SERIALIZATION(MapProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, OptionalProto);
  nb::enum_<OptionalProto::DataType>(nb_OptionalProto, "DataType", nb::is_arithmetic())
      .value("UNDEFINED", OptionalProto::DataType::UNDEFINED)
      .value("TENSOR", OptionalProto::DataType::TENSOR)
      .value("SPARSE_TENSOR", OptionalProto::DataType::SPARSE_TENSOR)
      .value("SEQUENCE", OptionalProto::DataType::SEQUENCE)
      .value("MAP", OptionalProto::DataType::MAP)
      .value("OPTIONAL", OptionalProto::DataType::OPTIONAL)
      .export_values();
  nb_OptionalProto.SHORTEN_CODE(OptionalProto::DataType, UNDEFINED)
      .SHORTEN_CODE(OptionalProto::DataType, TENSOR)
      .SHORTEN_CODE(OptionalProto::DataType, SPARSE_TENSOR)
      .SHORTEN_CODE(OptionalProto::DataType, SEQUENCE)
      .SHORTEN_CODE(OptionalProto::DataType, MAP)
      .SHORTEN_CODE(OptionalProto::DataType, OPTIONAL);
  nb_OptionalProto.PYFIELD_STR(OptionalProto, name)
      .def_prop_rw(
          "elem_type",
          [](const OptionalProto &self) -> OptionalProto::DataType { return self.elem_type_; },
          [](OptionalProto &self, nb::object obj) {
            if (nb::isinstance<nb::int_>(obj)) {
              self.elem_type_ = static_cast<OptionalProto::DataType>(nb::cast<int>(obj));
            } else {
              self.elem_type_ = nb::cast<OptionalProto::DataType>(obj);
            }
          },
          OptionalProto::DOC_elem_type)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, tensor_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, sparse_tensor_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, sequence_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, map_value)
      .PYFIELD_OPTIONAL_PROTO(OptionalProto, optional_value)
      .def("HasField", [](const OptionalProto &self, const std::string &field_name) {
        if (self.has_tensor_value() && field_name == "tensor_value")
          return true;
        if (self.has_sparse_tensor_value() && field_name == "sparse_tensor_value")
          return true;
        if (self.has_sequence_value() && field_name == "sequence_value")
          return true;
        if (self.has_map_value() && field_name == "map_value")
          return true;
        if (self.has_optional_value() && field_name == "optional_value")
          return true;
        return false;
      });
  PYADD_PROTO_SERIALIZATION(OptionalProto);
}
