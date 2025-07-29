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

template <typename T> void RepeatedProtoField<T>::clear() {
  for (auto &p : values) {
    p.reset();
  }
  values.clear();
}

template <typename T> inline T &RepeatedProtoField<T>::operator[](size_t index) {
  return *values[index];
}

template <typename T> inline const T &RepeatedProtoField<T>::operator[](size_t index) const {
  return *values[index];
}

template <typename T> void RepeatedProtoField<T>::push_back(const T &v) { values.push_back(std::make_unique<T>()); }

template <typename T> void RepeatedProtoField<T>::extend(const std::vector<T> &v) {
  for (const auto &item : v) {
    values.push_back(std::make_unique<T>(item));
  }
}

template <typename T> void RepeatedProtoField<T>::extend(const std::vector<T *> &v) {
  for (const auto &item : v) {
    EXT_ENFORCE(item != nullptr, "Cannot extend RepeatedProtoField with a null pointer.");
    values.push_back(std::make_unique<T>(*item));
  }
}

template <typename T> void RepeatedProtoField<T>::extend(const std::vector<T *> &&v) {
  for (const auto &item : v) {
    EXT_ENFORCE(item != nullptr, "Cannot extend RepeatedProtoField with a null pointer.");
    values.push_back(std::move(item));
  }
  v.clear();
}

template <typename T> void RepeatedProtoField<T>::extend(const RepeatedProtoField<T> &v) {
  for (const auto &item : v.values) {
    values.push_back(std::make_unique<T>(*item));
  }
}

template <typename T> void RepeatedProtoField<T>::extend(const RepeatedProtoField<T> &&v) {
  for (const auto &item : v.values) {
    values.push_back(std::move(item));
  }
  v.values.clear();
}

template <typename T> T &RepeatedProtoField<T>::add() {
  values.push_back(std::make_unique<T>());
  return *values.back();
}

template <typename T> T &RepeatedProtoField<T>::back() {
  EXT_ENFORCE(!values.empty(), "Cannot call back() on an empty RepeatedField.");
  return *values.back();
}

template <typename T> std::vector<std::string> RepeatedProtoField<T>::SerializeToVectorString() const {
  std::vector<std::string> rows{"["};
  for (const auto &p : values) {
    std::vector<std::string> r = p->SerializeToVectorString();
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
  reset();
  if (v.has_value()) {
    set_empty_value();
    StringWriteStream stream;
    (*v).SerializeToStream(stream);
    StringStream rstream(stream.data(), stream.size());
    value->ParseFromStream(rstream);
  }
  return *this;
}

} // namespace utils
} // namespace onnx2
