#pragma once

#include "helpers.h"
#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

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


inline void _ThrowOnError_(OrtStatus *ort_status, const char *filename,
                           int line, const OrtApi &api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(
            orthelpers::MakeString("error: onnxruntime(", code, "), ", message,
                                   "\n    ", filename, ":", line));
      }
    }
  }
}

#define ThrowOnError(api, ort_status)                                          \
  _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

inline std::string KernelInfoGetOptionalAttributeString(
    const OrtApi &api, const OrtKernelInfo *info, const char *name,
    const std::string &default_value) {
  size_t size = 0;
  std::string out;

  OrtStatus *status =
      api.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return default_value;
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  out.resize(size);
  ThrowOnError(api,
               api.KernelInfoGetAttribute_string(info, name, &out[0], &size));
  out.resize(size - 1); // remove the terminating character '\0'
  return out;
}

template <typename T>
inline OrtStatus *KernelInfoGetAttributeApi(const OrtApi &api,
                                            const OrtKernelInfo *info,
                                            const char *name, T &out);

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<int64_t>(const OrtApi &api, const OrtKernelInfo *info,
                                   const char *name, int64_t &out) {
  return api.KernelInfoGetAttribute_int64(info, name, &out);
}

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<float>(const OrtApi &api, const OrtKernelInfo *info,
                                 const char *name, float &out) {
  return api.KernelInfoGetAttribute_float(info, name, &out);
}

template <typename T>
inline T KernelInfoGetOptionalAttribute(const OrtApi &api,
                                        const OrtKernelInfo *info,
                                        const char *name, T default_value) {
  T out;
  OrtStatus *status = KernelInfoGetAttributeApi<T>(api, info, name, out);

  if (status == nullptr) {
    return out;
  }
  OrtErrorCode code = api.GetErrorCode(status);
  if (code == ORT_FAIL) {
    api.ReleaseStatus(status);
    return default_value;
  }

  ThrowOnError(api, status);
  return default_value;
}

inline bool KernelInfoGetOptionalAttributeInt64AsBool(const OrtApi &api,
                                                      const OrtKernelInfo *info,
                                                      const char *name,
                                                      bool default_value) {
  int64_t value = KernelInfoGetOptionalAttribute<int64_t>(
      api, info, name, default_value ? 1 : 0);
  return value == 1;
}

} // namespace ortops
