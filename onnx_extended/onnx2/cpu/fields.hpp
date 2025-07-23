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

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const T &) {
  EXT_THROW("Assignment is now allowed yet for an option field.");
}

} // namespace utils
} // namespace onnx2
