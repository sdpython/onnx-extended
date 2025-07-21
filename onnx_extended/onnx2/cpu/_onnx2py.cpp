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

  py::class_<onnx2::OperatorSetIdProto>(m, "OperatorSetIdProto",
                                        "OperatorSetIdProto, opset definition")
      .def(py::init<>())
      .FIELD(OperatorSetIdProto, domain)
      .FIELD(OperatorSetIdProto, version)
      .ADD_PROTO_SERIALIZATION(OperatorSetIdProto);

  py::class_<onnx2::TensorShapeProto::Dimension>(m, "Dimension",
                                                 "Dimension, an integer value or a string")
      .def(py::init<>())
      .FIELD(TensorShapeProto::Dimension, dim_value)
      .FIELD(TensorShapeProto::Dimension, dim_param)
      .FIELD(TensorShapeProto::Dimension, dim_value)
      .FIELD(TensorShapeProto::Dimension, denotation)
      .ADD_PROTO_SERIALIZATION(TensorShapeProto::Dimension);

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
      .FIELD(TensorProto, data_type)
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
