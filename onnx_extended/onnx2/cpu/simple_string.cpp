#include "simple_string.h"
#include <sstream>

namespace onnx2 {
namespace utils {

bool RefString::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool RefString::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  if (data() == other.data())
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool RefString::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator!=(const std::string &other) const { return !(*this == other); }
bool RefString::operator!=(const String &other) const { return !(*this == other); }
bool RefString::operator!=(const RefString &other) const { return !(*this == other); }
bool RefString::operator!=(const char *other) const { return !(*this == other); }

std::string RefString::as_string() const {
  if (empty())
    return std::string();
  return std::string(data(), size());
}

String::String(const RefString &s) : size_(s.size()) {
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
}

String::String(const String &s) : size_(s.size()) {
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
}

String::String(const char *ptr, size_t size) {
  if (ptr == nullptr || size == 0 || (size == 1 && ptr[0] == 0)) {
    ptr_ = nullptr;
    size_ = 0;
  } else if (ptr[size - 1] == 0) {
    ptr_ = new char[size - 1];
    memcpy(ptr_, ptr, size - 1);
    size_ = size - 1;
  } else {
    ptr_ = new char[size];
    memcpy(ptr_, ptr, size);
    size_ = size;
  }
}

String::String(const std::string &s) : size_(s.size()) {
  ptr_ = new char[s.size()];
  memcpy(ptr_, s.c_str(), size_);
}

String &String::operator=(const char *s) {
  clear();
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
  clear();
  size_ = s.size();
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
  return *this;
}

String &String::operator=(const String &s) {
  clear();
  size_ = s.size();
  if (size_ > 0) {
    ptr_ = new char[size_];
    memcpy(ptr_, s.data(), size_);
  } else {
    ptr_ = nullptr;
  }
  return *this;
}

String &String::operator=(const std::string &s) {
  clear();
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

bool String::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool String::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator!=(const std::string &other) const { return !(*this == other); }
bool String::operator!=(const String &other) const { return !(*this == other); }
bool String::operator!=(const RefString &other) const { return !(*this == other); }
bool String::operator!=(const char *other) const { return !(*this == other); }

std::string String::as_string() const {
  if (empty())
    return std::string();
  return std::string(data(), size());
}

std::string join_string(const std::vector<std::string> &elements, const char *delimiter) {
  std::stringstream oss;
  auto it = elements.begin();
  if (it != elements.end()) {
    oss << *it;
    ++it;
  }
  while (it != elements.end()) {
    oss << delimiter << *it;
    ++it;
  }
  return oss.str();
}

} // namespace utils
} // namespace onnx2
