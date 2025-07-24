#pragma once

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

#define BEGIN_PROTO(cls)                                                                       \
  class cls : public Message {                                                                 \
  public:                                                                                      \
    static const char *DOC;                                                                    \
    inline cls() {}

#define BEGIN_PROTO_NOINIT(cls)                                                                \
  class cls : public Message {                                                                 \
  public:                                                                                      \
    static const char *DOC;

#define END_PROTO()                                                                            \
  SERIALIZATION_METHOD()                                                                       \
  }                                                                                            \
  ;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif

#define FDEC_1 0
#define FDEC_2 1
#define FDEC_3 2
#define FDEC(x) FDEC_##x
#define FCONCAT(a, b) a##b
#define FEXPAND_CONCAT(a, b) FCONCAT(a, b)

#define FIELD(type, name, order)                                                               \
public:                                                                                        \
  inline type &name() { return name##_; }                                                      \
  inline const type &name() const { return name##_; }                                          \
  inline void set_##name(const type &v) { name##_ = v; }                                       \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  type name##_;                                                                                \
  using name##_t = type;

#define FIELD_ONEOF_BEGIN(varname)                                                             \
public:                                                                                        \
  union {

#define FIELD_ONEOF_0(type, name) type name##_;
#define FIELD_ONEOF_0_STRING(name) std::string *name##_;

#define FIELD_ONEOF_MIDDLE(varname)                                                            \
  }                                                                                            \
  ;                                                                                            \
  int32_t varname##_filled_order;                                                              \
  inline void clear_##varname##_0() {}

#define FIELD_ONEOF_1(varname, type, name, order)                                              \
  using name##_t = type;                                                                       \
  inline type &name() { return name##_; }                                                      \
  inline const type &name() const { return name##_; }                                          \
  inline void set_##name(const type &v) {                                                      \
    name##_ = v;                                                                               \
    varname##_filled_order = order;                                                            \
  }                                                                                            \
  inline bool has_##name() const { return varname##_filled_order == order; }                   \
  inline void clear_##name() {                                                                 \
    if (has_##name()) {                                                                        \
      clear_one_of(name##_);                                                                   \
      varname##_filled_order = 0;                                                              \
    }                                                                                          \
  }                                                                                            \
  inline void clear_##varname##_##order() {                                                    \
    clear_##name();                                                                            \
    FEXPAND_CONCAT(clear_##varname##_, FDEC(order))();                                         \
  }                                                                                            \
  inline int order_##name() const { return order; }

#define FIELD_ONEOF_1_STRING(varname, name, order)                                             \
  using name##_t = std::string;                                                                \
  inline type &name() { return has_##name() ? *name##_ : std::string(); }                      \
  inline const type &name() const { return has_##name() ? *name##_ : std::string(); }          \
  inline void set_##name(const type &v) {                                                      \
    if (has_##name())                                                                          \
      *name##_ = v;                                                                            \
    else                                                                                       \
      name##_ = new std::string(v);                                                            \
    varname##_filled_order = order;                                                            \
  }                                                                                            \
  inline bool has_##name() const { return varname##_filled_order == order; }                   \
  inline void clear_##name() {                                                                 \
    if (has_##name()) {                                                                        \
      if (&&name##_ != nullptr) {                                                              \
        delete name##_;                                                                        \
        name##_ = nullptr;                                                                     \
      }                                                                                        \
      varname##_filled_order = 0;                                                              \
    }                                                                                          \
  }                                                                                            \
  inline void clear_##varname##_##order() {                                                    \
    clear_##name();                                                                            \
    FEXPAND_CONCAT(clear_##varname##_, FDEC(order))();                                         \
  }                                                                                            \
  inline int order_##name() const { return order; }

#define FIELD_ONEOF_END(varname, n)                                                            \
  inline void clear_##varname() { clear_##varname##_##n(); };

#define FIELD_REPEATED(type, name, order)                                                      \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline void add_##name(type &&v) { name##_.emplace_back(v); }                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  inline bool packed_##name() const { return false; }                                          \
  utils::RepeatedField<type> name##_;                                                          \
  using name##_t = type;

#define FIELD_REPEATED_PACKED(type, name, order)                                               \
public:                                                                                        \
  inline utils::RepeatedField<type> &name() { return name##_; }                                \
  inline void add_##name(type &&v) { name##_.emplace_back(v); }                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }          \
  inline int order_##name() const { return order; }                                            \
  inline bool packed_##name() const { return true; }                                           \
  utils::RepeatedField<type> name##_;                                                          \
  using name##_t = type;

#define FIELD_OPTIONAL(type, name, order)                                                      \
public:                                                                                        \
  inline type &name() {                                                                        \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return *name##_;                                                                           \
  }                                                                                            \
  inline const type &name() const {                                                            \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return *name##_;                                                                           \
  }                                                                                            \
  inline utils::OptionalField<type> &name##_optional() {                                       \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return name##_;                                                                            \
  }                                                                                            \
  inline const utils::OptionalField<type> &name##_optional() const {                           \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");            \
    return name##_;                                                                            \
  }                                                                                            \
  inline void set_##name(const type &v) { name##_ = v; }                                       \
  inline void reset_##name() { name##_.reset(); }                                              \
  inline bool has_##name() const { return _has_field_(name##_); }                              \
  inline int order_##name() const { return order; }                                            \
  utils::OptionalField<type> name##_;                                                          \
  using name##_t = type;

namespace onnx2 {

using utils::offset_t;

template <typename T> inline void clear_one_of(T &) {}
template <> inline void clear_one_of(std::string &s) { s.clear(); }

template <typename T> inline bool _has_field_(const T &) { return true; }
template <> inline bool _has_field_(const std::string &field) { return !field.empty(); }
template <> inline bool _has_field_(const utils::OptionalField<uint64_t> &field) {
  return field.has_value();
}
template <> inline bool _has_field_(const std::vector<uint8_t> &field) {
  return !field.empty();
}

class Message {
public:
  inline Message() {}
  inline bool operator==(const Message &) const {
    EXT_THROW("operator == not implemented for a Message");
  }
};

} // namespace onnx2