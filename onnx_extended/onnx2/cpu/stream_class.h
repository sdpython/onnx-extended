#pragma once

#include "simple_string.h"
#include "stream.h"

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

#define BEGIN_PROTO(cls, doc)                                                                  \
  class cls : public Message {                                                                 \
  public:                                                                                      \
    static inline constexpr const char *DOC = doc;                                             \
    explicit inline cls() {}

#define BEGIN_PROTO_NOINIT(cls, doc)                                                           \
  class cls : public Message {                                                                 \
  public:                                                                                      \
    static inline constexpr const char *DOC = doc;

#define END_PROTO()                                                                            \
  SERIALIZATION_METHOD()                                                                       \
  }                                                                                            \
  ;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif

#define FIELD(type, name, order, doc)                                                          \
public:                                                                                        \
  inline type &name() { return name##_; }                                                      \
  inline const type &name() const { return name##_; }                                          \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline void set_##name(const type &v) { name##_ = v; }                                       \
  inline int order_##name() const { return order; }                                            \
  static inline constexpr const char *DOC_##name = doc;                                        \
  type name##_;                                                                                \
  using name##_t = type;

#define FIELD_DEFAULT(type, name, order, default_value, doc)                                   \
public:                                                                                        \
  inline type &name() { return name##_; }                                                      \
  inline const type &name() const { return name##_; }                                          \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline void set_##name(const type &v) { name##_ = v; }                                       \
  inline int order_##name() const { return order; }                                            \
  static inline constexpr const char *DOC_##name = doc;                                        \
  type name##_ = default_value;                                                                \
  using name##_t = type;

#define FIELD_STR(name, order, doc)                                                            \
  FIELD(utils::String, name, order, doc)                                                       \
  inline void set_##name(const std::string &v) { name##_ = v; }                                \
  inline void set_##name(const utils::RefString &v) { name##_ = v; }

#define FIELD_REPEATED(type, name, order, doc)                                                 \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline type &add_##name() { return name##_.add(); }                                          \
  inline type &add_##name(type &&v) {                                                          \
    name##_.emplace_back(v);                                                                   \
    return name##_.back();                                                                     \
  }                                                                                            \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  static inline constexpr const char *DOC_##name = doc;                                        \
  inline bool packed_##name() const { return false; }                                          \
  utils::RepeatedField<type> name##_;                                                          \
  using name##_t = type;

#define FIELD_REPEATED_PACKED(type, name, order, doc)                                          \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline type &add_##name() { return name##_.add(); }                                          \
  inline type &add_##name(const type &v) {                                                     \
    name##_.push_back(v);                                                                      \
    return name##_.back();                                                                     \
  }                                                                                            \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  static inline constexpr const char *DOC_##name = doc;                                        \
  inline bool packed_##name() const { return true; }                                           \
  utils::RepeatedField<type> name##_;                                                          \
  using name##_t = type;

#define _FIELD_OPTIONAL(type, name, order, doc)                                                \
public:                                                                                        \
  inline type &name() {                                                                        \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return *name##_;                                                                           \
  }                                                                                            \
  inline const type &name() const {                                                            \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return *name##_;                                                                           \
  }                                                                                            \
  inline utils::OptionalField<type> &name##_optional() { return name##_; }                     \
  inline const utils::OptionalField<type> &name##_optional() const {                           \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return name##_;                                                                            \
  }                                                                                            \
  inline type &add_##name() {                                                                  \
    name##_.set_empty_value();                                                                 \
    return *name##_;                                                                           \
  }                                                                                            \
  inline void set_##name(const type &v) { name##_ = v; }                                       \
  inline void reset_##name() { name##_.reset(); }                                              \
  inline bool has_##name() const { return name##_.has_value(); }                               \
  inline int order_##name() const { return order; }                                            \
  static inline constexpr const char *DOC_##name = doc;                                        \
  utils::OptionalField<type> name##_;                                                          \
  using name##_t = type;

#define FIELD_OPTIONAL(type, name, order, doc)                                                 \
  _FIELD_OPTIONAL(type, name, order, doc)                                                      \
  inline bool has_oneof_##name() const { return has_##name(); }

#define FIELD_OPTIONAL_ONEOF(type, name, order, oneof, doc)                                    \
  _FIELD_OPTIONAL(type, name, order, doc)                                                      \
  inline bool has_oneof_##name() const { return has_##oneof(); }

namespace onnx2 {

using utils::offset_t;

template <typename T> inline bool _has_field_(const T &) { return true; }
template <> inline bool _has_field_(const utils::String &field) { return !field.empty(); }
template <> inline bool _has_field_(const std::vector<uint8_t> &field) {
  return !field.empty();
}

class Message {
public:
  explicit inline Message() {}
  inline bool operator==(const Message &) const {
    EXT_THROW("operator == not implemented for a Message");
  }
};

} // namespace onnx2