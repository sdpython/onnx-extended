#pragma once

#include "stream_class.h"
#include "stream_class_print.hpp"
#include "stream_class_read.hpp"
#include "stream_class_size.hpp"
#include "stream_class_write.hpp"

////////////////
// macro helpers
////////////////

#define NAME_EXIST_VALUE(name) name_exist_value(_name_##name, has_##name(), ptr_##name())

#define IMPLEMENT_PROTO(cls)                                                                           \
  void cls::CopyFrom(const cls &proto) {                                                               \
    utils::StringWriteStream stream;                                                                   \
    SerializeOptions opts;                                                                             \
    proto.SerializeToStream(stream, opts);                                                             \
    utils::StringStream read_stream(stream.data(), stream.size());                                     \
    ParseOptions ropts;                                                                                \
    ParseFromStream(read_stream, ropts);                                                               \
  }                                                                                                    \
  uint64_t cls::SerializeSize() const {                                                                \
    SerializeOptions opts;                                                                             \
    utils::StringWriteStream stream;                                                                   \
    return SerializeSize(stream, opts);                                                                \
  }                                                                                                    \
  void cls::ParseFromString(const std::string &raw) {                                                  \
    ParseOptions opts;                                                                                 \
    ParseFromString(raw, opts);                                                                        \
  }                                                                                                    \
  void cls::ParseFromString(const std::string &raw, ParseOptions &opts) {                              \
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());                                \
    onnx2::utils::StringStream st(ptr, raw.size());                                                    \
    if (opts.parallel)                                                                                 \
      st.StartThreadPool(opts.num_threads);                                                            \
    ParseFromStream(st, opts);                                                                         \
    if (opts.parallel)                                                                                 \
      st.WaitForDelayedBlock();                                                                        \
  }                                                                                                    \
  void cls::SerializeToString(std::string &out) const {                                                \
    SerializeOptions opts;                                                                             \
    SerializeToString(out, opts);                                                                      \
  }                                                                                                    \
  void cls::SerializeToString(std::string &out, SerializeOptions &opts) const {                        \
    onnx2::utils::StringWriteStream buf;                                                               \
    auto &opts_ref = opts;                                                                             \
    SerializeToStream(buf, opts_ref);                                                                  \
    out = std::string(reinterpret_cast<const char *>(buf.data()), buf.size());                         \
  }

///////////////////////
// macro serialize size
///////////////////////

#define SIZE_FIELD(size, options, stream, name)                                                        \
  if (has_##name()) {                                                                                  \
    size += size_field(stream, order_##name(), ref_##name(), options);                                 \
  }

#define SIZE_FIELD_NULL(size, options, stream, name)                                                   \
  if (!name##_.null()) {                                                                               \
    size += size_field(stream, order_##name(), ref_##name(), options);                                 \
  }

#define SIZE_FIELD_EMPTY(size, options, stream, name)                                                  \
  size += size_field(stream, order_##name(), ref_##name(), options);

#define SIZE_FIELD_LIMIT(size, options, stream, name)                                                  \
  if (has_##name()) {                                                                                  \
    size += size_field_limit(stream, order_##name(), ref_##name(), options);                           \
  }

#define SIZE_ENUM_FIELD(size, options, stream, name)                                                   \
  if (has_##name()) {                                                                                  \
    size += size_enum_field(stream, order_##name(), ref_##name(), options);                            \
  }

#define SIZE_REPEATED_FIELD(size, options, stream, name)                                               \
  if (has_##name()) {                                                                                  \
    size += size_repeated_field(stream, order_##name(), name##_, packed_##name(), options);            \
  }

#define SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, name)                                         \
  if (has_##name()) {                                                                                  \
    size += size_optional_proto_field(stream, order_##name(), name##_optional(), options);             \
  }

//////////////
// macro write
//////////////

#define WRITE_FIELD(options, stream, name)                                                             \
  if (has_##name()) {                                                                                  \
    write_field(stream, order_##name(), ref_##name(), options);                                        \
  }

#define WRITE_FIELD_NULL(options, stream, name)                                                        \
  if (!name##_.null()) {                                                                               \
    write_field(stream, order_##name(), ref_##name(), options);                                        \
  }

#define WRITE_FIELD_EMPTY(options, stream, name)                                                       \
  write_field(stream, order_##name(), ref_##name(), options);

#define WRITE_FIELD_LIMIT(options, stream, name)                                                       \
  if (has_##name()) {                                                                                  \
    write_field_limit(stream, order_##name(), ref_##name(), options);                                  \
  }

#define WRITE_ENUM_FIELD(options, stream, name)                                                        \
  if (has_##name()) {                                                                                  \
    write_enum_field(stream, order_##name(), ref_##name(), options);                                   \
  }

#define WRITE_REPEATED_FIELD(options, stream, name)                                                    \
  if (has_##name()) {                                                                                  \
    write_repeated_field(stream, order_##name(), name##_, packed_##name(), options);                   \
  }

#define WRITE_OPTIONAL_PROTO_FIELD(options, stream, name)                                              \
  if (has_##name()) {                                                                                  \
    write_optional_proto_field(stream, order_##name(), name##_optional(), options);                    \
  }

/////////////
// macro read
/////////////

#define READ_BEGIN(options, stream, cls)                                                               \
  DEBUG_PRINT("+ read begin " #cls)                                                                    \
  while (stream.NotEnd()) {                                                                            \
    utils::FieldNumber field_number = stream.next_field();                                             \
    DEBUG_PRINT2("  = field number ", field_number.string().c_str())                                   \
    if (field_number.field_number == 0) {                                                              \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);                \
    }

#define READ_END(options, stream, cls)                                                                 \
  else {                                                                                               \
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);             \
  }                                                                                                    \
  }                                                                                                    \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(options, stream, name)                                                              \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + field " #name)                                                                    \
    read_field(stream, field_number.wire_type, name##_, #name, options);                               \
    DEBUG_PRINT("  - field " #name)                                                                    \
  }

#define READ_FIELD_LIMIT_PARALLEL(options, stream, name)                                               \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + field " #name)                                                                    \
    read_field_limit_parallel(stream, field_number.wire_type, name##_, #name, options);                \
    DEBUG_PRINT("  - field " #name)                                                                    \
  }

#define READ_OPTIONAL_PROTO_FIELD(options, stream, name)                                               \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + optional field " #name)                                                           \
    read_optional_proto_field(stream, field_number.wire_type, name##_, #name, options);                \
    DEBUG_PRINT("  - optional field " #name)                                                           \
  }

#define READ_ENUM_FIELD(options, stream, name)                                                         \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + enum " #name)                                                                     \
    read_enum_field(stream, field_number.wire_type, name##_, #name, options);                          \
    DEBUG_PRINT("  - enum " #name)                                                                     \
  }

#define READ_OPTIONAL_ENUM_FIELD(options, stream, name)                                                \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + enum " #name)                                                                     \
    read_optional_enum_field(stream, field_number.wire_type, name##_, #name, options);                 \
    DEBUG_PRINT("  - enum " #name)                                                                     \
  }

#define READ_REPEATED_FIELD(options, stream, name)                                                     \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                            \
    DEBUG_PRINT("  + repeat " #name)                                                                   \
    read_repeated_field(stream, field_number.wire_type, name##_, #name, packed_##name(), options);     \
    DEBUG_PRINT("  - repeat " #name)                                                                   \
  }

using namespace onnx_extended_helpers;

namespace onnx2 {} // namespace onnx2
