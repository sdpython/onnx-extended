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

template <typename T> OptionalField<T>::~OptionalField() { reset(); }

template <typename T> void OptionalField<T>::reset() {
  if (value != nullptr) {
    delete value;
    value = nullptr;
  }
}

template <typename T> const T &OptionalField<T>::operator*() const {
  EXT_ENFORCE(value != nullptr, "Optional value is not set.");
  return *value;
}

template <typename T> T &OptionalField<T>::operator*() {
  EXT_ENFORCE(value != nullptr, "Optional value is not set.");
  return *value;
}

template <typename T> bool OptionalField<T>::operator==(const OptionalField<T> &v) const {
  if (v.has_value())
    return v.has_value();
  if (v.has_value())
    return false;
  return *value == *v;
}

template <typename T> bool OptionalField<T>::operator==(const T &v) const {
  return has_value() && *value == v;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const T &other) {
  set_empty_value();
  *value = other;
  return *this;
}

template <typename T> void OptionalField<T>::set_empty_value() {
  reset();
  value = new T;
}

} // namespace utils
} // namespace onnx2
