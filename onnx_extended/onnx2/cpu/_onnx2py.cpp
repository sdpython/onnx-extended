#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx2.h"

namespace py = pybind11;
using namespace validation;

#define ADD_PROTO_SERIALIZATION(cls)                                                           \
  .def(                                                                                        \
      "ParseFromString",                                                                       \
      [](onnx2::cls &self, py::bytes data) {                                                   \
        std::string raw = data;                                                                \
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());                    \
        onnx2::utils::StringStream st(ptr, raw.size());                                        \
        self.ParseFromString(st);                                                              \
      },                                                                                       \
      "Parses a sequence of bytes to fill this instance")                                      \
      .def(                                                                                    \
          "SerializeToString",                                                                 \
          [](onnx2::cls &self) {                                                               \
            onnx2::utils::StringWriteStream buf;                                               \
            self.SerializeToString(buf);                                                       \
            return py::bytes(reinterpret_cast<const char *>(buf.data()), buf.size());          \
          },                                                                                   \
          "Serialize into a sequence of bytes.")

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

  py::class_<onnx2::StringStringEntryProto>(m, "StringStringEntryProto",
                                            "StringStringEntryProto, a key, a value")
      .def(py::init<>())
      .def_readwrite("key", &onnx2::StringStringEntryProto::key, "key")
      .def_readwrite("value", &onnx2::StringStringEntryProto::value, "value")
          ADD_PROTO_SERIALIZATION(StringStringEntryProto);

  py::class_<onnx2::TensorShapeProto::Dimension>(m, "Dimension",
                                                 "Dimension, an integer value or a string")
      .def(py::init<>())
      .def_readwrite("dim_value", &onnx2::TensorShapeProto::Dimension::dim_value, "dim_value")
      .def_readwrite("dim_param", &onnx2::TensorShapeProto::Dimension::dim_param, "dim_param")
      .def_readwrite("denotation", &onnx2::TensorShapeProto::Dimension::denotation,
                     "denotation") ADD_PROTO_SERIALIZATION(TensorShapeProto::Dimension);

  py::class_<onnx2::TensorShapeProto>(m, "TensorShapeProto",
                                      "TensorShapeProto, multiple DimProto")
      .def(py::init<>())
      .def_readwrite("dim", &onnx2::TensorShapeProto::dim, "dim")
          ADD_PROTO_SERIALIZATION(TensorShapeProto);

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
      .def_readwrite("dims", &onnx2::TensorProto::dims, "shape")
      .def_readwrite("data_type", &onnx2::TensorProto::data_type, "data type")
      .def_readwrite("name", &onnx2::TensorProto::name, "name")
      .def_readwrite("doc_string", &onnx2::TensorProto::doc_string, "doc_string")
      .def_readwrite("metadata_props", &onnx2::TensorProto::metadata_props, "metadata_props")
      .def_property(
          "raw_data",
          [](const onnx2::TensorProto &self) -> py::bytes {
            return py::bytes(reinterpret_cast<const char *>(self.raw_data.data()),
                             self.raw_data.size());
          },
          [](onnx2::TensorProto &self, py::bytes data) {
            std::string raw = data;
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            self.raw_data.resize(raw.size());
            memcpy(self.raw_data.data(), ptr, raw.size());
          },
          "raw_data")
      .def(
          "ParseFromString",
          [](onnx2::TensorProto &self, py::bytes data) {
            std::string raw = data;
            onnx2::utils::StringStream st(reinterpret_cast<const uint8_t *>(raw.data()),
                                          raw.size());
            self.ParseFromString(st);
          },
          "Parses a sequence of bytes to fill this instance");
}
