#include "cpu/c_op_tfidf_vectorizer_.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace onnx_c_ops {

template <typename Sequence>
inline py::array_t<typename Sequence::value_type, py::array::c_style | py::array::forcecast>
as_pyarray(const std::vector<int64_t> &shape, Sequence &&seq) {
  Sequence *seq_ptr = new Sequence(std::move(seq));
  auto capsule = py::capsule(seq_ptr, [](void *p) { delete reinterpret_cast<Sequence *>(p); });
  return py::array_t<typename Sequence::value_type, py::array::c_style | py::array::forcecast>(
      shape, seq_ptr->data(), capsule);
}

class PyRuntimeTfIdfVectorizer {
public:
  PyRuntimeTfIdfVectorizer() : tfidf_() {}

  void Init(int max_gram_length, int max_skip_count, int min_gram_length,
            const std::string &mode, const std::vector<int64_t> &ngram_counts,
            const std::vector<int64_t> &ngram_indexes, const std::vector<int64_t> &pool_int64s,
            const std::vector<float> &weights, bool sparse) {
    tfidf_.Init(max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts,
                ngram_indexes, pool_int64s, weights, sparse);
  }
  ~PyRuntimeTfIdfVectorizer() {}

  py::array_t<float, py::array::c_style | py::array::forcecast>
  Compute(py::array_t<int64_t, py::array::c_style | py::array::forcecast> X) const {
    std::vector<int64_t> input_shape;
    arrayshape2vector(input_shape, X);
    std::size_t total_dims = static_cast<std::size_t>(flattened_dimension(input_shape));
    const int64_t *p = X.data();
    onnx_c_ops::RuntimeTfIdfVectorizer<float>::span_type_int64 sp{p, total_dims};
    std::vector<int64_t> output_dims;
    std::vector<float> out;
    tfidf_.Compute(input_shape, sp,
                   [&dims = output_dims, &o = out](const std::vector<int64_t> &out_dims)
                       -> onnx_c_ops::RuntimeTfIdfVectorizer<float>::span_type {
                     dims = out_dims;
                     int64_t total = flattened_dimension(out_dims);
                     o.resize(total);
                     return onnx_c_ops::RuntimeTfIdfVectorizer<float>::span_type{
                         o.data(), static_cast<size_t>(total)};
                   });
    return as_pyarray<std::vector<float>>(output_dims, std::move(out));
  }

private:
  onnx_c_ops::RuntimeTfIdfVectorizer<float> tfidf_;
};

} // namespace onnx_c_ops

using namespace onnx_c_ops;

PYBIND11_MODULE(c_op_tfidf_vectorizer_py_, m) {
  m.doc() =
#if defined(__APPLE__)
      "Implements runtime for operator TfIdfVectorizer."
#else
      R"pbdoc(Implements runtime for operator TfIdfVectorizer. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
#endif
      ;

  py::class_<PyRuntimeTfIdfVectorizer> cli(
      m, "PyRuntimeTfIdfVectorizer",
      R"pbdoc(Implements runtime for operator TfIdfVectorizer. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc>`_
in :epkg:`onnxruntime`. Supports Int only.)pbdoc");

  cli.def(py::init<>());
  cli.def("init", &PyRuntimeTfIdfVectorizer::Init, "Initializes PyRuntimeTfIdfVectorizer.");
  cli.def("compute", &PyRuntimeTfIdfVectorizer::Compute, "Computes PyRuntimeTfIdfVectorizer.");
}
