#pragma once

#include "onnx_extended_helpers.h"
#include <cstddef>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace onnx2 {
namespace utils {

template <typename T> std::vector<std::string> RepeatedField<T>::SerializeToVectorString() const {
  std::vector<std::string> rows{"["};
  for (const auto &p : values) {
    std::vector<std::string> r = p.SerializeToVectorString();
    for (size_t i = 0; i < r.size(); ++i) {
      if (i + 1 == r.size()) {
        rows.push_back(onnx_extended_helpers::MakeString("  ", r[i], ","));
      } else {
        rows.push_back(onnx_extended_helpers::MakeString("  ", r[i]));
      }
    }
  }
  rows.push_back("],");
  return rows;
}

template <typename T> OptionalField<T>::~OptionalField() { reset(); }

template <typename T> void OptionalField<T>::reset() {
  if (value != nullptr) {
    delete value;
    value = nullptr;
  }
}

template <typename T> void OptionalField<T>::set_empty_value() {
  reset();
  value = new T;
}

template <typename T> T &OptionalField<T>::operator*() {
  EXT_ENFORCE(has_value(), "Optional field has no value.");
  return *value;
}

template <typename T> const T &OptionalField<T>::operator*() const {
  EXT_ENFORCE(has_value(), "Optional field has no value.");
  return *value;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const T &v) {
  // We make a copy.
  set_empty_value();
  StringWriteStream stream;
  v.SerializeToStream(stream);
  StringStream rstream(stream.data(), stream.size());
  value->ParseFromStream(rstream);
  return *this;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const OptionalField<T> &v) {
  // We make a copy.
  set_empty_value();
  if (v.has_value()) {
    StringWriteStream stream;
    (*v).SerializeToStream(stream);
    StringStream rstream(stream.data(), stream.size());
    value->ParseFromStream(rstream);
  }
  return *this;
}

} // namespace utils
} // namespace onnx2
