#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpu_fpemu.hpp"
#include "murmur_hash3.h"
#include "onnx2.h"
#include "speed_metrics.h"
#include "vector_sparse.h"

namespace py = pybind11;
using namespace validation;

PYBIND11_MODULE(_validation, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations."
#else
      R"pbdoc(C++ experimental implementations.)pbdoc"
#endif
      ;

  m.def("benchmark_cache", &benchmark_cache, py::arg("size"), py::arg("verbose") = true,
        R"pbdoc(Runs a benchmark to measure the cache performance.
The function measures the time for N random accesses in array of size N
and returns the time divided by N.
It copies random elements taken from the array size to random
position in another of the same size. It does that *size* times
and return the average time per move.
See example :ref:`l-example-bench-cpu`.

:param size: array size
:return: average time per move

The function measures the time spent in this loop.

::

    for (int64_t i = 0; i < arr_size; ++i) {
        // index k will jump forth and back, to generate cache misses
        int64_t k = (i / 2) + (i % 2) * arr_size / 2;
        arr_b[k] = arr_a[k] + 1;
     }

The code is `benchmark_cache
<https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/validation/cpu/speed_metrics.cpp#L17>`_.
)pbdoc");

  py::class_<ElementTime> clf(m, "ElementTime");
  clf.def(py::init<int64_t, int64_t, double>());
  clf.def_readwrite("trial", &ElementTime::trial);
  clf.def_readwrite("row", &ElementTime::row);
  clf.def_readwrite("time", &ElementTime::time);

  m.def("benchmark_cache_tree", &benchmark_cache_tree, py::arg("n_rows") = 100000,
        py::arg("n_features") = 50, py::arg("n_trees") = 200, py::arg("tree_size") = 4096,
        py::arg("max_depth") = 10, py::arg("search_step") = 64,
        R"pbdoc(Simulates the prediction of a random forest.
Returns the time taken by every rows for a function doing
random addition between an element from the same short buffer and
another one taken from a list of trees.
See example :ref:`l-example-bench-cpu`.

:param n_rows: number of rows of the whole batch size
:param n_features: number of features
:param n_trees: number of trees
:param tree_size: size of a tree (= number of nodes * sizeof(node) / sizeof(float))
:param max_depth: depth of a tree
:param search_step: evaluate every...
:return: array of time take for every row

The code is `benchmark_cache_tree
<https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/validation/cpu/speed_metrics.cpp#L50>`_
)pbdoc");

  m.def(
      "murmurhash3_bytes_s32",
      [](const std::string &key, uint32_t seed) -> int32_t {
        int32_t out;
        sklearn::MurmurHash3_x86_32(key.data(), key.size(), seed, &out);
        return out;
      },
      py::arg("key"), py::arg("seed") = 0,
      R"pbdoc(Calls murmurhash3_bytes_s32 from scikit-learn.

:param key: string
:param seed: unsigned integer
:return: hash
)pbdoc");

#if defined(__SSSE3__)

  m.def(
      "has_sse3", []() -> bool { return true; },
      R"pbdoc(Tells if SSE3 instructions are available. 
They are needed to convert floart to half and half to float.)pbdoc");

  m.def("double2float_rn", &cpu_fpemu::__double2float_rn, py::arg("d"),
        R"pbdoc(Converts a double into float.)pbdoc");

  m.def("float2half_rn", &cpu_fpemu::__float2half_rn, py::arg("d"),
        R"pbdoc(Converts a float into half represented as an unsigned short.)pbdoc");

  m.def("half2float", &cpu_fpemu::__half2float, py::arg("d"),
        R"pbdoc(Converts a half represented as an unsigned short into float.)pbdoc");

#else

  m.def(
      "has_sse3", []() -> bool { return false; },
      R"pbdoc(Tells if SSE3 instructions are available. 
They are needed to convert floart to half and half to float.)pbdoc");

#endif

  m.def("sparse_struct_to_dense", &sparse_struct_to_dense, py::arg("v"),
        R"pbdoc(Converts a sparse structure stored in a float tensor
into a dense vector.)pbdoc");

  m.def("sparse_struct_to_maps", &sparse_struct_to_maps, py::arg("v"),
        R"pbdoc(Converts a sparse structure stored in a float tensor
into a list of dictionaries. The sparse tensor needs to be 2D.)pbdoc");

  m.def("dense_to_sparse_struct", &dense_to_sparse_struct, py::arg("v"),
        R"pbdoc(Converts a dense float tensor into a sparse structure
stored in a float tensor.)pbdoc");

  m.def("sparse_struct_to_csr", &sparse_struct_to_csr, py::arg("v"),
        R"pbdoc(Returns the position of the first elements of each row.
The array has n+1 elements if n is the number of non null elements.
The second array stores the column index for every element.)pbdoc");

  m.def("sparse_struct_indices_values", &sparse_struct_indices_values, py::arg("v"),
        R"pbdoc(Returns the indices and the values from a sparse structure
stored in a float tensor.)pbdoc");

  m.def(
      "evaluate_sparse",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> values_array, int random,
         int ntimes, int repeat, int test) -> std::vector<std::tuple<double, double, double>> {
        EXT_ENFORCE(values_array.size() > 0, "Input tensor is empty.");
        EXT_ENFORCE(random > 0, "random is null.");
        EXT_ENFORCE(ntimes > 0, "ntimes is null.");
        EXT_ENFORCE(repeat > 0, "repeat is null.");
        const float *values = values_array.data(0);
        std::vector<int64_t> dims(values_array.ndim());
        for (std::size_t i = 0; i < dims.size(); ++i)
          dims[i] = (int64_t)values_array.shape(i);
        EXT_ENFORCE(dims.size() == 2, "2D tensor is expected.");
        return evaluate_sparse(values, dims[0], dims[1], random, ntimes, repeat, test);
      },
      py::arg("tensor"), py::arg("n_random"), py::arg("n_times"), py::arg("repeat"),
      py::arg("dense"),
      R"pbdoc(Returns computation time about random access to features dense or sparse, 
initialization time, loop time, sum of the element from the array based on random indices.
The goal is to evaluate whether or not it is faster to switch to a dense representation
or to keep the sparse representation to do random access to the structures.

:param tensor: dense tensor to access
:param n_random: number of random access
:param n_times: number of times to do n_random random accesses
:param repeat: number of times to repeat the measure
:param dense: if true, measure the conversion from sparse and then operate
      random access on a dense structure, if false, operator random access directly
      into the sparse structures
:return: 3-tuple, intialization time (conversion to dense, or sparse initialization),
      measure of the random accesses, control sum
)pbdoc");

  m.def(
      "utils_onnx2_read_varint64",
      [](py::bytes data) -> py::tuple {
        std::string raw = data;
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
        offset_t pos = 0;
        int64_t value = onnx2::utils::readVarint64(ptr, pos, raw.size());
        return py::make_tuple(value, pos);
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
      .def(
          "ParseFromString",
          [](onnx2::StringStringEntryProto &self, py::bytes data) {
            std::string raw = data;
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            size_t pos = 0;
            self.ParseFromString(ptr, pos, raw.size());
          },
          "Parses a sequence of bytes to fill this instance");

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
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            size_t pos = 0;
            self.ParseFromString(ptr, pos, raw.size());
          },
          "Parses a sequence of bytes to fill this instance");
}
