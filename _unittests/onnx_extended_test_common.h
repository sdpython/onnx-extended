#pragma once

#include "test_constants.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if __cplusplus >= 201703L
#include <filesystem>
#endif
#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

#ifdef _WIN32
typedef std::wstring std_string_type;
#else
typedef std::string std_string_type;
#endif

#if !defined(EXPECT_TRUE)
#define EXPECT_TRUE(condition)                                                                 \
  {                                                                                            \
    if (!(condition)) {                                                                        \
      throw std::runtime_error(                                                                \
          onnx_extended_helpers::MakeString(__FILE__, ":", __LINE__, " in ", __FUNCTION__));   \
    }                                                                                          \
  }
#endif

#if !defined(EXPECT_EQ)
#define EXPECT_EQ(a, b)                                                                        \
  {                                                                                            \
    if (a != b) {                                                                              \
      throw std::runtime_error(onnx_extended_helpers::MakeString(                              \
          __FILE__, ":", __LINE__, " in ", __FUNCTION__, "\n", "a != b"));                     \
    }                                                                                          \
  }
#endif

#if !defined(EXPECT_NE)
#define EXPECT_NE(a, b)                                                                        \
  {                                                                                            \
    if (a == b) {                                                                              \
      throw std::runtime_error(onnx_extended_helpers::MakeString(                              \
          __FILE__, ":", __LINE__, " in ", __FUNCTION__, "\n", "a == b"));                     \
    }                                                                                          \
  }
#endif

template <typename T> bool check_equal(int n, T *pa, T *pb) {
  for (int i = 0; i < n; ++i) {
    if (pa[i] != pb[i])
      return false;
  }
  return true;
}

#if !defined(EXPECT_EQ_VECTOR)
#define EXPECT_EQ_VECTOR(n, pa, pb) EXPECT_TRUE(check_equal(n, pa, pb))
#endif

template <typename T> bool check_almost_equal(int n, T *pa, T *pb, T precision = 1e-5) {
  for (int i = 0; i < n; ++i) {
    if (pa[i] == pb[i])
      continue;
    T d = pa[i] - pb[i];
    d = d > 0 ? d : -d;
    if (d > precision)
      return false;
  }
  return true;
}

#if !defined(EXPECT_ALMOST_VECTOR)
#define EXPECT_ALMOST_VECTOR(n, pa, pb, prec) EXPECT_TRUE(check_almost_equal(n, pa, pb, prec))
#endif

inline std_string_type to_std_string_path(const char *path) {
#if ((!defined(PYTHON_MANYLINUX) || !PYTHON_MANYLINUX) && __cplusplus >= 201703L)
#ifdef _WIN32
  std::wstring model(path);
  return model;
#else
  std::string model(path);
  return model;
#endif
#else
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
  std::wstring model(cvt.from_bytes(std::string(path)));
  return model;
#else
  std::string model(path);
  return model;
#endif
#endif
}

inline std_string_type get_data_path(const char *path) {
#if ((!defined(PYTHON_MANYLINUX) || !PYTHON_MANYLINUX) && __cplusplus >= 201703L)
  std::filesystem::path cwd = TEST_FOLDER;
#ifdef _WIN32
  std::wstring model = (cwd / path).wstring();
  return model;
#else
  std::string model = (cwd / path).string();
  return model;
#endif
#else
  std::string cwd = TEST_FOLDER;
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
  std::wstring model(cvt.from_bytes(cwd + std::string("/") + std::string(path)));
  return model;
#else
  std::string model = cwd + std::string("/") + std::string(path);
  return model;
#endif
#endif
}