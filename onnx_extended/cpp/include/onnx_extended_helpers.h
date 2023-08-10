#pragma once

#include <algorithm>
#include <float.h>
#include <iostream> // cout
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>

namespace onnx_extended_helpers {

inline std::vector<std::string> SplitString(const std::string &input,
                                            char delimiter) {
  std::vector<std::string> parts;
  std::string::size_type start = 0;
  std::string::size_type end = input.find(delimiter);

  while (end != std::string::npos) {
    parts.push_back(input.substr(start, end - start));
    start = end + 1;
    end = input.find(delimiter, start);
  }

  parts.push_back(input.substr(start));
  return parts;
}

inline void MakeStringInternal(std::ostringstream &ss) noexcept {}

template <typename T>
inline void MakeStringInternal(std::ostringstream &ss, const T &t) noexcept {
  ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int32_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint32_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int64_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint64_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<int16_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream &ss,
                               const std::vector<uint16_t> &t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream &ss, const T &t,
                               const Args &...args) noexcept {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args> inline std::string MakeString(const Args &...args) {
  std::ostringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

#if !defined(_THROW_DEFINED)
#define EXT_THROW(...)                                                         \
  throw std::runtime_error(onnx_extended_helpers::MakeString(                  \
      "[onnx-extended] ", onnx_extended_helpers::MakeString(__VA_ARGS__)));
#define _THROW_DEFINED
#endif

#if !defined(_ENFORCE_DEFINED)
#define EXT_ENFORCE(cond, ...)                                                 \
  if (!(cond))                                                                 \
    throw std::runtime_error(onnx_extended_helpers::MakeString(                \
        "`", #cond, "` failed. ",                                              \
        onnx_extended_helpers::MakeString(                                     \
            "[onnx-extended] ",                                                \
            onnx_extended_helpers::MakeString(__VA_ARGS__))));
#define _ENFORCE_DEFINED
#endif

} // namespace onnx_extended_helpers
