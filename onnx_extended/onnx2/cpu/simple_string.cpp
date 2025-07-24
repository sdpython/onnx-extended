#include "simple_string.h"

namespace onnx2 {
namespace utils {

String::String(RefString s) : size_(s.size()) {
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
}

String::String(const char *ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    ptr_ = nullptr;
    size_ = 0;
  } else if (ptr[size - 1] == 0) {
    ptr_ = new char[size];
    memcpy(ptr_, ptr, size);
  } else {
    ptr_ = new char[size + 1];
    memcpy(ptr_, ptr, size + 1);
  }
}

String::String(const std::string &s) : size_(s.size()) {
  ptr_ = new char[s.size() + 1];
  memcpy(ptr_, s.c_str(), size_);
  ptr_[s.size()] = 0;
}

String &String::operator=(const char *s) {
  if (s == nullptr || s[0] == 0) {
    size_ = 0;
    ptr_ = nullptr;
  } else {
    size_ = 0;
    for (; s[size_] != 0; ++size_)
      ;
    ptr_ = new char[size_];
    memcpy(ptr_, s, size_);
  }
  return *this;
}

String &String::operator=(const RefString &s) {
  size_ = s.size();
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
  return *this;
}

bool String::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool String::operator==(const String &other) const {
  if (size_ != other.size_)
    return false;
  if (size_ == 0)
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool String::operator==(RefString other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool String::operator!=(const String &other) const { return !(*this == other); }
bool String::operator!=(RefString other) const { return !(*this == other); }
bool String::operator!=(const char *other) const { return !(*this == other); }

} // namespace utils
} // namespace onnx2
