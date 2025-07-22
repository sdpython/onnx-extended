#pragma once

#include "stream.h"
#include <optional>

#define FIELD_VARINT 0
// #define FIELD_FIXED64 1
#define FIELD_FIXED_SIZE 2
// #define FIELD_FIXED32 5

#define SERIALIZATION_METHOD()                                                                 \
  inline void ParseFromString(const std::string &raw) {                                        \
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());                        \
    onnx2::utils::StringStream st(ptr, raw.size());                                            \
    ParseFromStream(st);                                                                       \
  }                                                                                            \
  inline void SerializeToString(std::string &out) const {                                      \
    onnx2::utils::StringWriteStream buf;                                                       \
    SerializeToStream(buf);                                                                    \
    out = std::string(reinterpret_cast<const char *>(buf.data()), buf.size());                 \
  }                                                                                            \
  void ParseFromStream(utils::BinaryStream &stream);                                           \
  void SerializeToStream(utils::BinaryWriteStream &stream) const;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif
#define FIELD(type, name, order)                                                               \
public:                                                                                        \
  inline type &name() { return name##_; }                                                      \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  type name##_;

#define FIELD_REPEATED(type, name, order)                                                      \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  inline bool packed_##name() const { return false; }                                          \
  utils::RepeatedField<type> name##_;

#define FIELD_REPEATED_PACKED(type, name, order)                                               \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  inline bool packed_##name() const { return true; }                                           \
  utils::RepeatedField<type> name##_;

#define FIELD_OPTIONAL(type, name, order)                                                      \
public:                                                                                        \
  inline type &name() {                                                                        \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return *name##_;                                                                           \
  }                                                                                            \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  utils::OptionalField<type> name##_;

namespace onnx2 {

using utils::offset_t;

template <typename T> inline bool _has_field_(const T &) { return true; }
template <> inline bool _has_field_(const std::string &field) { return !field.empty(); }
template <> inline bool _has_field_(const std::optional<uint64_t> &field) {
  return field.has_value();
}
template <> inline bool _has_field_(const std::vector<uint8_t> &field) {
  return !field.empty();
}

class Message {
public:
  inline Message() {}
};

} // namespace onnx2