#pragma once

#include "onnx_extended_helpers.h"
#include <cstring>
#include <string>

namespace onnx2 {
namespace utils {

class RefString {
private:
  const char *ptr_;
  size_t size_;

public:
  inline RefString(const RefString &copy) : ptr_(copy.ptr_), size_(copy.size_) {}
  inline RefString(const char *ptr, size_t size) : ptr_(ptr), size_(size) {}
  inline RefString &operator=(const RefString &) = default;
  inline size_t size() const { return size_; }
  inline const char *c_str() const { return ptr_; }
  inline const char *data() const { return ptr_; }
  inline bool empty() const { return size_ == 0; }
  inline char operator[](size_t i) const { return ptr_[i]; }
};

class String {
private:
  char *ptr_;
  size_t size_;

public:
  inline ~String() {
    if (ptr_ != nullptr)
      delete ptr_;
  }
  inline String() : ptr_(nullptr), size_(0) {}
  String(RefString s);
  String(const char *ptr, size_t size);
  String(const std::string &s);
  inline size_t size() const { return size_; }
  inline const char *c_str() const { return ptr_; }
  inline const char *data() const { return ptr_; }
  inline bool empty() const { return size_ == 0; }
  inline char operator[](size_t i) const { return ptr_[i]; }
  String &operator=(const char *s);
  String &operator=(const RefString &s);
  bool operator==(const String &other) const;
  bool operator==(RefString other) const;
  bool operator==(const char *other) const;
  bool operator!=(const String &other) const;
  bool operator!=(RefString other) const;
  bool operator!=(const char *other) const;
};

} // namespace utils
} // namespace onnx2
