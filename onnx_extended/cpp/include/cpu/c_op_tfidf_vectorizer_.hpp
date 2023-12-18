// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc.
#pragma once

#include "common/c_op_common_parallel.hpp"
#include "common/c_op_helpers.h"
#include "common/sparse_tensor.h"

#include "onnx_extended_helpers.h"
#include <cstring>
#include <functional>
#include <map>
#if __cpluscplus >= 202002L
#include <span>
#else
#include "common/simple_span.h"
#endif
#include <sstream>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

using namespace onnx_extended_helpers;

namespace onnx_c_ops {

// NgrampPart implements a Trie like structure
// for a unigram (1) it would insert into a root map with a valid id.
// for (1,2,3) node 2 would be a child of 1 but have id == 0
// because (1,2) does not exists. Node 3 would have a valid id.
template <class T> class NgramPart;

template <> class NgramPart<int64_t>;

using NgramPartInt = NgramPart<int64_t>;

class IntMap : public std::unordered_map<int64_t, NgramPartInt *> {
public:
  inline IntMap() : std::unordered_map<int64_t, NgramPartInt *>() {}
  inline ~IntMap();
  inline std::string to_string(const std::string &indent = "") const;
};

template <> class NgramPart<int64_t> {
public:
  size_t id_; // 0 - means no entry, search for a bigger N
  IntMap leafs_;
  inline NgramPart(size_t id) : id_(id) {}
  inline ~NgramPart() {}
  inline std::string to_string(const std::string &indent = "") const {
    if (leafs_.size() == 0)
      return MakeString("NGramPart(", static_cast<int64_t>(id_), ")");
    return MakeString("NGramPart(", static_cast<int64_t>(id_), ", ", leafs_.to_string(indent),
                      ")");
  }
};

inline IntMap::~IntMap() {
  for (auto it = begin(); it != end(); ++it)
    delete it->second;
}

inline std::string IntMap::to_string(const std::string &indent) const {
  std::vector<std::string> rows;
  rows.reserve(size());
  rows.push_back("{");
  int irow = 0;
  for (auto pair = cbegin(); pair != cend(); ++pair, ++irow) {
    auto v = pair->second->to_string(indent + "  ");
    if (irow == 0)
      rows.push_back(MakeString(indent, pair->first, "=", v));
    else
      rows.push_back(MakeString(indent, pair->first, "=", v, ","));
  }
  rows.push_back("}");
  std::stringstream ss;
  for (auto line : rows) {
    ss << line << "\n";
  }
  return ss.str();
}

// Returns next ngram_id
template <class K, class ForwardIter, class Map>
inline size_t PopulateGrams(ForwardIter first, size_t ngrams, size_t ngram_size,
                            size_t ngram_id, Map &c) {
  for (; ngrams > 0; --ngrams) {
    size_t n = 1;
    Map *m = &c;
    while (true) {
      auto p = m->emplace(*first, new NgramPart<int64_t>(0));
      ++first;
      if (n == ngram_size) {
        p.first->second->id_ = ngram_id;
        ++ngram_id;
        break;
      }
      ++n;
      m = &p.first->second->leafs_;
    }
  }
  return ngram_id;
}

inline const void *AdvanceElementPtr(const void *p, size_t elements, size_t element_size) {
  return reinterpret_cast<const uint8_t *>(p) + elements * element_size;
}

// The weighting criteria.
// "TF"(term frequency),
//    the counts are propagated to output
// "IDF"(inverse document frequency),
//    all the counts larger than 1
//    would be truncated to 1 and the i-th element
//    in weights would be used to scale (by multiplication)
//    the count of the i-th n-gram in pool
// "TFIDF" (the combination of TF and IDF).
//  counts are scaled by the associated values in the weights attribute.

enum WeightingCriteria { kNone = 0, kTF = 1, kIDF = 2, kTFIDF = 3 };

template <typename T> class RuntimeTfIdfVectorizer {

public:
#if __cpluscplus >= 202002L
  typedef std::span<T> span_type;
  typedef std::span<const int64_t> span_type_int64;
#else
  typedef std_::span<T> span_type;
  typedef std_::span<const int64_t> span_type_int64;
#endif

  RuntimeTfIdfVectorizer() {
    weighting_criteria_ = WeightingCriteria::kNone;
    max_gram_length_ = 0;
    min_gram_length_ = 0;
    max_skip_count_ = 0;
    output_size_ = 0;
  }

  void Init(int max_gram_length, int max_skip_count, int min_gram_length,
            const std::string &mode, const std::vector<int64_t> &ngram_counts,
            const std::vector<int64_t> &ngram_indexes, const std::vector<int64_t> &pool_int64s,
            const std::vector<T> &weights, bool sparse) {
    if (mode == "TF")
      weighting_criteria_ = kTF;
    else if (mode == "IDF")
      weighting_criteria_ = kIDF;
    else if (mode == "TFIDF")
      weighting_criteria_ = kTFIDF;

    min_gram_length_ = min_gram_length;
    max_gram_length_ = max_gram_length;
    max_skip_count_ = max_skip_count;
    ngram_counts_ = ngram_counts;
    ngram_indexes_ = ngram_indexes;
    sparse_ = sparse;

    auto greatest_hit = std::max_element(ngram_indexes_.cbegin(), ngram_indexes_.cend());
    output_size_ = *greatest_hit + 1;

    weights_ = weights;
    pool_int64s_ = pool_int64s;

    const auto total_items = pool_int64s.size();
    size_t ngram_id = 1; // start with 1, 0 - means no n-gram
    // Load into dictionary only required gram sizes
    size_t ngram_size = 1;
    for (size_t i = 0; i < ngram_counts_.size(); ++i) {

      size_t start_idx = ngram_counts_[i];
      size_t end_idx = ((i + 1) < ngram_counts_.size()) ? ngram_counts_[i + 1] : total_items;
      auto items = end_idx - start_idx;
      if (items > 0) {
        auto ngrams = items / ngram_size;
        if ((int)ngram_size >= min_gram_length && (int)ngram_size <= max_gram_length)
          ngram_id = PopulateGrams<int64_t>(pool_int64s.begin() + start_idx, ngrams, ngram_size,
                                            ngram_id, int64_map_);
        else
          ngram_id += ngrams;
      }
      ++ngram_size;
    }
  }

  ~RuntimeTfIdfVectorizer() {}

  void Compute(const std::vector<int64_t> &input_dims, const span_type_int64 &X,
               std::function<span_type(const std::vector<int64_t> &)> alloc) const {
    const size_t total_items = flattened_dimension(input_dims);

    size_t num_rows = 0;
    size_t B = 0;
    size_t C = 0;
    if (input_dims.empty()) {
      num_rows = 1;
      C = 1;
      if (total_items != 1)
        throw std::invalid_argument("Unexpected total of items.");
    } else if (input_dims.size() == 1) {
      num_rows = 1;
      C = input_dims[0];
    } else if (input_dims.size() == 2) {
      B = input_dims[0];
      C = input_dims[1];
      num_rows = B;
      if (B < 1)
        throw std::invalid_argument(
            "Input shape must have either [C] or [B,C] dimensions with B > 0.");
    } else
      throw std::invalid_argument(
          "Input shape must have either [C] or [B,C] dimensions with B > 0.");

    if (num_rows * C != total_items)
      throw std::invalid_argument("Unexpected total of items.");
    // Frequency holder allocate [B..output_size_]
    // and init all to zero

    std::vector<int64_t> output_dims;
    if (B == 0) {
      output_dims.push_back(output_size_);
      B = 1; // For use in the loops below
    } else {
      output_dims.push_back(B);
      output_dims.push_back(output_size_);
    }

    if (sparse_) {
      if (total_items == 0 || int64_map_.empty()) {
        // TfidfVectorizer may receive an empty input when it follows a
        // Tokenizer (for example for a string containing only stopwords).
        // TfidfVectorizer returns a zero tensor of shape
        // {b_dim, output_size} when b_dim is the number of received
        // observations and output_size the is the maximum value in
        // ngram_indexes attribute plus 1.
        std::vector<T> output;
        onnx_sparse::sparse_struct::copy(output_dims, std::vector<uint32_t>(), std::vector<T>(),
                                         output);
        span_type out = alloc(std::vector<int64_t>{static_cast<int64_t>(output.size())});
        std::memcpy(out.data(), output.data(), output.size() * sizeof(T));
        return;
      }
      ComputeSparse(X, output_dims, num_rows, C, alloc);
    } else {
      if (total_items == 0 || int64_map_.empty()) {
        // TfidfVectorizer may receive an empty input when it follows a
        // Tokenizer (for example for a string containing only stopwords).
        // TfidfVectorizer returns a zero tensor of shape
        // {b_dim, output_size} when b_dim is the number of received
        // observations and output_size the is the maximum value in
        // ngram_indexes attribute plus 1.
        span_type out = alloc(output_dims);
        std::memset(out.data(), 0, out.size() * sizeof(T));
        return;
      }
      ComputeDense(X, output_dims, num_rows, C, alloc);
    }
  }

private:
  void ComputeDense(const span_type_int64 &X, const std::vector<int64_t> &output_dims,
                    const size_t num_rows, const size_t C,
                    std::function<span_type(const std::vector<int64_t> &)> alloc) const {

    span_type out = alloc(output_dims);

    std::function<void(size_t, float *)> fn_weight;
    // can be parallelized.
    size_t n_threads = omp_get_max_threads();
    size_t n_per_threads = std::min(static_cast<size_t>(128),
                                    std::max(num_rows / n_threads / 2, static_cast<size_t>(1)));

    auto &w = weights_;

    switch (weighting_criteria_) {
    case kTF:
      fn_weight = [](size_t i, float *out) { out[i] += 1.0f; };
      break;
    case kIDF:
      if (!w.empty()) {
        fn_weight = [&w](size_t i, float *out) { out[i] = w[i]; };
      } else {
        fn_weight = [](size_t i, float *out) { out[i] = 1.0f; };
      }
      break;
    case kTFIDF:
      if (!w.empty()) {
        fn_weight = [&w](size_t i, float *out) { out[i] += w[i]; };
      } else {
        fn_weight = [](size_t i, float *out) { out[i] += 1.0f; };
      }
      break;
    case kNone: // fall-through
    default:
      EXT_THROW("Unexpected weight type configuration for TfIdfVectorizer.");
    }

    TryBatchParallelFor2i(
        n_threads, n_per_threads, num_rows,
        [this, X, C, &out, &fn_weight](int, ptrdiff_t row_start, ptrdiff_t row_end) {
          auto begin = out.data() + row_start * this->output_size_;
          auto end = out.data() + row_end * this->output_size_;
          std::fill(begin, end, static_cast<T>(0));
          for (auto row_num = row_start; row_num < row_end;
               ++row_num, begin += this->output_size_) {
            ComputeImpl(X.data() + row_num * C, C, begin, fn_weight);
          }
        });
  }

  void ComputeSparse(const span_type_int64 &X, const std::vector<int64_t> &output_dims,
                     const size_t num_rows, const size_t C,
                     std::function<span_type(const std::vector<int64_t> &)> alloc) const {

    std::function<void(uint32_t, std::map<uint32_t, T> & out)> fn_weight;
    // can be parallelized.
    size_t n_threads = omp_get_max_threads();
    size_t n_per_threads = std::min(static_cast<size_t>(128),
                                    std::max(num_rows / n_threads / 2, static_cast<size_t>(1)));

    auto &w = weights_;

    switch (weighting_criteria_) {
    case kTF:
      fn_weight = [](uint32_t i, std::map<uint32_t, T> &out) {
        auto it = out.find(i);
        if (it == out.end())
          out[i] = 1;
        else
          it->second += 1.0f;
      };
      break;
    case kIDF:
      if (!w.empty()) {
        fn_weight = [&w](uint32_t i, std::map<uint32_t, T> &out) { out[i] = w[i]; };
      } else {
        fn_weight = [](uint32_t i, std::map<uint32_t, T> &out) { out[i] = 1.0f; };
      }
      break;
    case kTFIDF:
      if (!w.empty()) {
        fn_weight = [&w](uint32_t i, std::map<uint32_t, T> &out) {
          auto it = out.find(i);
          if (it == out.end())
            out[i] = w[i];
          else
            it->second += w[i];
        };
      } else {
        fn_weight = [](uint32_t i, std::map<uint32_t, T> &out) {
          auto it = out.find(i);
          if (it == out.end())
            out[i] = 1.0f;
          else
            it->second += 1.0f;
        };
      }
      break;
    case kNone: // fall-through
    default:
      EXT_THROW("Unexpected weight type configuration for TfIdfVectorizer.");
    }

    std::vector<std::vector<uint32_t>> indices(num_rows);
    std::vector<std::vector<T>> values(num_rows);

    TryBatchParallelFor2i(n_threads, n_per_threads, num_rows,
                          [this, X, C, &indices, &values, &fn_weight](int, ptrdiff_t row_start,
                                                                      ptrdiff_t row_end) {
                            for (auto row_num = row_start; row_num < row_end; ++row_num) {
                              // indices should be ordered
                              std::map<uint32_t, T> out;
                              ComputeImpl(X.data() + row_num * C, C, out, fn_weight);
                              indices[row_num].resize(out.size());
                              values[row_num].resize(out.size());
                              auto &this_indices = indices[row_num];
                              auto &this_values = values[row_num];
                              size_t i = 0;
                              for (auto it = out.begin(); it != out.end(); ++it, ++i) {
                                this_indices[i] =
                                    it->first + static_cast<uint32_t>(row_num) *
                                                    static_cast<uint32_t>(this->output_size_);
                                this_values[i] = it->second;
                              }
                            }
                          });

    int64_t total = 0;
    for (auto &it : indices)
      total += static_cast<int64_t>(it.size());

    onnx_sparse::sparse_struct sp;
    sp.set(output_dims, total, onnx_sparse::CTypeToElementType<T>().onnx_type());

    std::vector<int64_t> sparse_dims{static_cast<int64_t>(sp.size_float())};
    span_type out = alloc(sparse_dims);
    std::memcpy(static_cast<void *>(out.data()), static_cast<void *>(&sp), sizeof(sp) - 4);
    onnx_sparse::sparse_struct *spmoved = (onnx_sparse::sparse_struct *)(out.data());
    uint32_t *p_indices = spmoved->indices();
    T *p_values = spmoved->values();
    for (size_t i = 0; i < indices.size(); ++i) {
      std::memcpy(p_indices, indices[i].data(), indices[i].size() * sizeof(uint32_t));
      std::memcpy(p_values, values[i].data(), values[i].size() * sizeof(T));
      p_indices += indices[i].size();
      p_values += values[i].size();
    }
  }

  template <typename F, typename C>
  void ComputeImpl(const int64_t *X_data, size_t row_size, C &out, F &fn_weight) const {

    const auto elem_size = sizeof(int64_t);

    const void *row_begin = AdvanceElementPtr((void *)X_data, 0, elem_size);
    const void *const row_end = AdvanceElementPtr(row_begin, row_size, elem_size);

    const auto max_gram_length = max_gram_length_;
    const auto max_skip_distance = max_skip_count_ + 1; // Convert to distance
    auto start_ngram_size = min_gram_length_;

    for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
      auto ngram_start = row_begin;
      auto const ngram_row_end = row_end;

      while (ngram_start < ngram_row_end) {
        // We went far enough so no n-grams of any size can be gathered
        auto at_least_this =
            AdvanceElementPtr(ngram_start, skip_distance * (start_ngram_size - 1), elem_size);
        if (at_least_this >= ngram_row_end) {
          break;
        }

        auto ngram_item = ngram_start;
        const IntMap *int_map = &int64_map_;
        for (auto ngram_size = 1;
             !int_map->empty() && ngram_size <= max_gram_length && ngram_item < ngram_row_end;
             ++ngram_size,
                  ngram_item = AdvanceElementPtr(ngram_item, skip_distance, elem_size)) {
          int64_t val = *reinterpret_cast<const int64_t *>(ngram_item);
          auto hit = int_map->find(val);
          if (hit == int_map->end())
            break;
          if (ngram_size >= start_ngram_size && hit->second->id_ != 0) {
            fn_weight(OutputIdToIncrement(hit->second->id_), out);
          }
          int_map = &hit->second->leafs_;
        }
        // Sliding window shift
        ngram_start = AdvanceElementPtr(ngram_start, 1, elem_size);
      }
      // We count UniGrams only once since they are not affected
      // by skip distance
      if (start_ngram_size == 1 && ++start_ngram_size > max_gram_length)
        break;
    }
  }

  inline int64_t OutputIdToIncrement(size_t ngram_id) const {
    return ngram_indexes_[--ngram_id];
  }

private:
  WeightingCriteria weighting_criteria_;
  int64_t max_gram_length_;
  int64_t min_gram_length_;
  int64_t max_skip_count_;
  bool sparse_;
  std::vector<int64_t> ngram_counts_;
  std::vector<int64_t> ngram_indexes_;
  std::vector<T> weights_;
  std::vector<int64_t> pool_int64s_;
  IntMap int64_map_;
  size_t output_size_ = 0;
};

} // namespace onnx_c_ops
