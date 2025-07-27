#pragma once

#include "onnx_extended_helpers.h"
#include "simple_string.h"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace onnx2 {
namespace utils {

template <typename T> class RepeatedField {
public:
  explicit inline RepeatedField() {}
  inline void reserve(size_t n) { values.reserve(n); }
  inline void clear() { values.clear(); }
  inline bool empty() const { return values.empty(); }
  inline size_t size() const { return values.size(); }
  inline T &operator[](size_t index) { return values[index]; }
  inline const T &operator[](size_t index) const { return values[index]; }
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", static_cast<int>(step));
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", static_cast<int>(start));
    EXT_ENFORCE(stop == size(),
                "remove_range not implemented for stop=", static_cast<int>(stop),
                " and size=", static_cast<int>(size()));
    clear();
  }
  inline void extend(const std::vector<T> &v) {
    values.insert(values.end(), v.begin(), v.end());
  }
  inline void push_back(const T &v) { values.push_back(v); }
  inline void extend(const RepeatedField<T> &v) {
    values.insert(values.end(), v.begin(), v.end());
  }

  inline T &add() {
    values.emplace_back(T());
    return values.back();
  }

  inline T &back() { return values.back(); }

  inline std::vector<T>::iterator begin() { return values.begin(); }
  inline std::vector<T>::iterator end() { return values.end(); }
  inline std::vector<T>::const_iterator begin() const { return values.begin(); }
  inline std::vector<T>::const_iterator end() const { return values.end(); }
  template <class... Args> inline void emplace_back(Args &&...args) {
    values.emplace_back(std::forward<Args>(args)...);
  }
  std::vector<std::string> SerializeToVectorString() const;
  std::vector<T> values;
};

template <typename T> class OptionalField {
public:
  explicit inline OptionalField() : value(nullptr) {}
  explicit inline OptionalField(OptionalField<T> &&move) : value(move.value) {
    move.value = nullptr;
  }
  inline bool has_value() const { return value != nullptr; }
  ~OptionalField();
  void reset();
  T &operator*();
  const T &operator*() const;
  OptionalField<T> &operator=(const T &other);
  void set_empty_value();
  T *value;
};

template <typename T> class _OptionalField {
public:
  explicit inline _OptionalField() {}
  inline bool has_value() const { return value.has_value(); }
  inline void reset() { value.reset(); }
  inline const T &operator*() const { return *value; }
  inline T &operator*() { return *value; }
  inline bool operator==(const _OptionalField<T> &v) const { return value == v; }
  inline bool operator==(const T &v) const { return value == v; }
  inline _OptionalField<T> &operator=(const T &other) {
    value = other;
    return *this;
  }
  inline void set_empty_value() { value = static_cast<T>(0); }
  std::optional<T> value;
};

template <> class OptionalField<int64_t> : public _OptionalField<int64_t> {
public:
  explicit inline OptionalField() : _OptionalField<int64_t>() {}
  inline OptionalField<int64_t> &operator=(const int64_t &other) {
    value = other;
    return *this;
  }
};

template <> class OptionalField<int32_t> : public _OptionalField<int32_t> {
public:
  explicit inline OptionalField() : _OptionalField<int32_t>() {}
  inline OptionalField<int32_t> &operator=(const int32_t &other) {
    value = other;
    return *this;
  }
};

} // namespace utils
} // namespace onnx2
