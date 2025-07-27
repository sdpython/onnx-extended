#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
#include <gtest/gtest.h>

using namespace onnx2;

TEST(onnx2_string, RefString_Constructors) {
  utils::RefString original("test", 4);
  utils::RefString copied(original);
  EXPECT_EQ(copied.size(), 4);
  EXPECT_EQ(copied.data(), original.data());
  EXPECT_EQ(copied, original);

  const char *text = "hello";
  utils::RefString rs(text, 5);
  EXPECT_EQ(rs.size(), 5);
  EXPECT_EQ(rs.data(), text);
}

TEST(onnx2_string, RefString_Assignment) {
  utils::RefString a("abc", 3);
  utils::RefString b("xyz", 3);
  b = a;
  EXPECT_EQ(b.data(), a.data());
  EXPECT_EQ(b.size(), 3);

  utils::String s("def", 3);
  utils::RefString c("123", 3);
  c = s;
  EXPECT_EQ(c.data(), s.data());
  EXPECT_EQ(c.size(), 3);
}

TEST(onnx2_string, RefString_Methods) {
  utils::RefString a("hello", 5);
  EXPECT_EQ(a.size(), 5);
  EXPECT_EQ(a.c_str(), a.data());
  EXPECT_FALSE(a.empty());
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(a[0], 'h');
  EXPECT_EQ(a[4], 'o');
}

TEST(onnx2_string, RefString_Equality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  std::string e("test");
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty == nullptr);
}

TEST(onnx2_string, RefString_Inequality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  utils::String e("diff", 4);
  std::string f("test");
  std::string g("diff");
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);
  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);
  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx2_string, RefString_AsString) {
  utils::RefString a("hello", 5);
  std::string str = a.as_string();
  EXPECT_EQ(str, "hello");

  utils::RefString empty(nullptr, 0);
  std::string emptyStr = empty.as_string();
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx2_string, String_Constructors) {
  utils::String defaultStr;
  EXPECT_EQ(defaultStr.size(), 0);
  EXPECT_EQ(defaultStr.data(), nullptr);
  EXPECT_TRUE(defaultStr.empty());
  utils::RefString ref("test", 4);
  utils::String fromRef(ref);
  EXPECT_EQ(fromRef.size(), 4);
  EXPECT_NE(fromRef.data(), ref.data());
  EXPECT_EQ(fromRef, ref);

  utils::String fromCharPtr("hello", 5);
  EXPECT_EQ(fromCharPtr.size(), 5);
  EXPECT_EQ(fromCharPtr, "hello");

  utils::String withNull("abc\0", 4);
  EXPECT_EQ(withNull.size(), 3);
  EXPECT_EQ(withNull, "abc");

  std::string stdStr = "world";
  utils::String fromStdStr(stdStr);
  EXPECT_EQ(fromStdStr.size(), 5);
  EXPECT_EQ(fromStdStr, stdStr);
}

TEST(onnx2_string, String_Assignment) {
  utils::String s;

  s = "abc";
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, "abc");

  utils::RefString ref("def", 3);
  s = ref;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, ref);

  utils::String other("xyz", 3);
  s = other;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data());

  std::string stdStr = "hello";
  s = stdStr;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s, stdStr);
}

TEST(onnx2_string, String_Methods) {
  utils::String s("hello", 5);
  EXPECT_EQ(s.size(), 5);
  EXPECT_NE(s.data(), nullptr);
  EXPECT_FALSE(s.empty());
  utils::String empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(s[0], 'h');
  EXPECT_EQ(s[4], 'o');
  s.clear();
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s.data(), nullptr);
  EXPECT_TRUE(s.empty());
}

TEST(onnx2_string, String_Equality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  std::string e("test");

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::String empty;
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty == nullptr);
}

TEST(onnx2_string, String_Inequality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  utils::RefString e("diff", 4);
  std::string f("test");
  std::string g("diff");

  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);

  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);

  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx2_string, String_AsString) {
  utils::String a("hello", 5);
  std::string str = a.as_string();
  EXPECT_EQ(str, "hello");

  utils::String empty;
  std::string emptyStr = empty.as_string();
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx2_string, String_EdgeCases) {
  utils::String empty("");
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);

  utils::String null(nullptr, 0);
  EXPECT_TRUE(null.empty());
  EXPECT_EQ(null.size(), 0);

  utils::String withNulls("abc\0def", 7);
  EXPECT_EQ(withNulls.size(), 7);
  EXPECT_EQ(withNulls[3], '\0');
  EXPECT_EQ(withNulls[4], 'd');
}

TEST(onnx2_string, RefString) {
  utils::RefString a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
}

TEST(onnx2_string, String) {
  utils::String a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
  std::string s("iii");
  utils::String b(s);
  EXPECT_EQ(b.size(), 3);
  EXPECT_FALSE(b.empty());
  EXPECT_EQ(b, a);
  EXPECT_EQ(b, "iii");
}

TEST(onnx2_proto, TensorProtoName1) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.name().data(), nullptr);
  EXPECT_EQ(tp.name().size(), 0);
  std::string name("test");
  tp.name_ = name;
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  EXPECT_EQ(tp.order_name(), 8);
}

TEST(onnx2_proto, TensorProtoName2) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.name().data(), nullptr);
  EXPECT_EQ(tp.name().size(), 0);
  std::string name("test");
  tp.set_name(name);
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  std::string check = tp.name_.as_string();
  EXPECT_EQ(name, check);
  std::string check4 = tp.name().as_string();
  EXPECT_EQ(name, check4);
  name = "TEST2";
  tp.set_name(name);
  std::string check2 = tp.name_.as_string();
  EXPECT_EQ(name, check2);
}

TEST(onnx2_proto, TensorProtoNameStringToString1) {
  {
    TensorProto tp;
    tp.name_ = "test";
    if (tp.name().size() == 4) {
      TensorProto tp2;
      tp2.set_name(tp.name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp.name_.clear();
    }
    EXPECT_EQ(tp.name_.size(), 4);
    EXPECT_NE(tp.name_.data(), nullptr);
    EXPECT_EQ(tp.name_.data()[0], 't');
    EXPECT_EQ(tp.order_name(), 8);
    EXPECT_EQ(tp.name_, "test");
  }
}

TEST(onnx2_proto, TensorProtoNameStringToString2) {
  {
    TensorProto tp2;
    if (tp2.name().size() == 0) {
      TensorProto tp;
      tp.name_ = "test";
      tp2.set_name(tp.name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp2.name_.clear();
    }
    EXPECT_EQ(tp2.name_.size(), 4);
    EXPECT_NE(tp2.name_.data(), nullptr);
    EXPECT_EQ(tp2.name_.data()[0], 't');
    EXPECT_EQ(tp2.order_name(), 8);
    EXPECT_EQ(tp2.name_, "test");
  }
}

TEST(onnx2_proto, TensorProtoName00) { TensorProto tp; }
TEST(onnx2_proto, TensorProtoName01) {
  TensorProto tp;
  tp.set_name("rt");
}

TEST(onnx2_proto, serialization_StringStringEntryProto) {
  StringStringEntryProto proto;
  proto.key() = "key__";
  proto.value() = "value__";
  EXPECT_EQ(proto.key(), "key__");
  EXPECT_EQ(proto.value(), "value__");
  std::string serie;
  proto.SerializeToString(serie);
  StringStringEntryProto proto2;
  proto2.ParseFromString(serie);
  EXPECT_EQ(proto.key(), proto2.key());
  EXPECT_EQ(proto.value(), proto2.value());
  std::string serie2;
  proto2.SerializeToString(serie2);
  EXPECT_EQ(serie, serie2);
}

TEST(onnx2_proto, TensorShapeProto1) {
  TensorShapeProto shape;
  TensorShapeProto::Dimension &dim = shape.add_dim();
  dim.set_dim_value(5);
  TensorShapeProto::Dimension &dim2 = shape.dim().add();
  dim2.set_dim_param("dime");
  dim2.denotation() = "jj";
  EXPECT_EQ(shape.dim().size(), 2);
  EXPECT_EQ(shape.dim()[0].dim_value(), 5);
  EXPECT_EQ(shape.dim()[0].dim_param().size(), 0);
  EXPECT_EQ(shape.dim()[1].dim_param(), "dime");
  EXPECT_FALSE(shape.dim()[1].has_dim_value());
  EXPECT_EQ(shape.dim()[1].denotation(), "jj");
}

TEST(onnx2_stream, ZigZagEncoding) {
  int64_t original_values[] = {0, -1, 1, -2, 2, INT64_MAX, INT64_MIN};

  for (auto val : original_values) {
    uint64_t encoded = utils::encodeZigZag64(val);
    int64_t decoded = utils::decodeZigZag64(encoded);
    EXPECT_EQ(decoded, val) << "ZigZag encoding/decoding failed for value: " << val;
  }
}

TEST(onnx2_stream, FieldNumber) {
  utils::FieldNumber fn;
  fn.field_number = 5;
  fn.wire_type = 2;

  std::string str = fn.string();
  EXPECT_FALSE(str.empty());
  EXPECT_NE(str.find("field_number=5"), std::string::npos);
  EXPECT_NE(str.find("wire_type=2"), std::string::npos);
}

class onnx2_stream_2 : public ::testing::Test {
protected:
  void SetUp() override {
    data = {0x96, 0x01,
            // int64_t
            0x2A,
            // int32_t
            0x18,
            // float: 3.14
            0xC3, 0xF5, 0x48, 0x40,
            // double: 2.71828
            0x4D, 0xFB, 0x21, 0x09, 0x40, 0x05, 0x5D, 0x40,
            // field number: 10, wire_type: 2 -> (10 << 3) | 2 = 82
            0x52,
            // string length: 5
            0x05,
            // string "hello"
            'h', 'e', 'l', 'l', 'o'};

    stream = utils::StringStream(data.data(), data.size());
  }

  std::vector<uint8_t> data;
  utils::StringStream stream;
};

TEST_F(onnx2_stream_2, NextUInt64) {
  uint64_t value = stream.next_uint64();
  EXPECT_EQ(value, 150);
}

TEST_F(onnx2_stream_2, NextInt64) {
  stream.next_uint64();

  int64_t value = stream.next_int64();
  EXPECT_EQ(value, 42);
}

TEST_F(onnx2_stream_2, NextInt32) {
  stream.next_uint64();
  stream.next_int64();

  int32_t value = stream.next_int32();
  EXPECT_EQ(value, 24);
}

TEST_F(onnx2_stream_2, NextFloat) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  float value = stream.next_float();
  EXPECT_NEAR(value, 3.14f, 0.0001f);
}

TEST_F(onnx2_stream_2, NextField) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();

  utils::FieldNumber field = stream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);
}

TEST_F(onnx2_stream_2, NextString) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();
  stream.next_field();

  utils::RefString value = stream.next_string();
  EXPECT_EQ(value.size(), 5);
  EXPECT_EQ(value, "hello");
}

TEST_F(onnx2_stream_2, ReadBytes) {
  const uint8_t *bytes = stream.read_bytes(2);
  EXPECT_EQ(bytes[0], 0x96);
  EXPECT_EQ(bytes[1], 0x01);
}

TEST_F(onnx2_stream_2, CanRead) {
  stream.can_read(data.size(), "Test message");
  stream.read_bytes(10);
  stream.can_read(data.size() - 10, "Test message");
  EXPECT_THROW(stream.can_read(data.size(), "Test message"), std::runtime_error);
}

TEST_F(onnx2_stream_2, NotEnd) {
  EXPECT_TRUE(stream.not_end());
  stream.read_bytes(data.size() - 1);
  EXPECT_TRUE(stream.not_end());
  stream.read_bytes(1);
  EXPECT_FALSE(stream.not_end());
}

TEST_F(onnx2_stream_2, Tell) {
  EXPECT_EQ(stream.tell(), 0);

  stream.read_bytes(5);
  EXPECT_EQ(stream.tell(), 5);

  stream.read_bytes(10);
  EXPECT_EQ(stream.tell(), 15);
}

TEST(onnx2_stream, StringWriteStream) {
  utils::StringWriteStream stream;

  stream.write_variant_uint64(150);
  stream.write_int64(42);
  stream.write_int32(24);
  stream.write_float(3.14f);
  stream.write_double(2.71828);
  stream.write_field_header(10, 2);
  stream.write_string("hello");
  EXPECT_GT(stream.size(), 0);
  EXPECT_NE(stream.data(), nullptr);

  utils::StringStream readStream(stream.data(), stream.size());

  EXPECT_EQ(readStream.next_uint64(), 150);
  EXPECT_EQ(readStream.next_int64(), 42);
  EXPECT_EQ(readStream.next_int32(), 24);
  EXPECT_NEAR(readStream.next_float(), 3.14f, 0.0001f);
  EXPECT_NEAR(readStream.next_double(), 2.71828, 0.0001);

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);

  utils::RefString str = readStream.next_string();
  EXPECT_EQ(str, "hello");
}

TEST(onnx2_stream, StringWriteStreamStrings) {
  utils::StringWriteStream stream;

  std::string stdStr = "standard string";
  stream.write_string(stdStr);
  utils::String str("custom string", 13);
  stream.write_string(str);
  utils::RefString refStr("reference string", 16);
  stream.write_string(refStr);
  utils::StringStream readStream(stream.data(), stream.size());

  utils::RefString read1 = readStream.next_string();
  EXPECT_EQ(read1, "standard string");

  utils::RefString read2 = readStream.next_string();
  EXPECT_EQ(read2, "custom string");

  utils::RefString read3 = readStream.next_string();
  EXPECT_EQ(read3, "reference string");
}

TEST(onnx2_stream, BorrowedWriteStream) {
  std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
  utils::BorrowedWriteStream stream(data.data(), data.size());
  EXPECT_EQ(stream.size(), 5);
  EXPECT_EQ(stream.data(), data.data());
  EXPECT_THROW(stream.write_raw_bytes(nullptr, 0), std::runtime_error);
}

TEST(onnx2_stream, NestedStringWriteStreams) {
  utils::StringWriteStream innerStream;

  innerStream.write_string("inner data");
  utils::StringWriteStream outerStream;
  outerStream.write_field_header(15, 2);

  outerStream.write_string_stream(innerStream);

  utils::StringStream readStream(outerStream.data(), outerStream.size());

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 15);
  EXPECT_EQ(field.wire_type, 2);

  utils::StringStream innerReadStream;
  readStream.read_string_stream(innerReadStream);

  utils::RefString str = innerReadStream.next_string();
  EXPECT_EQ(str, "inner data");
}

// Tests pour les m�thodes next_packed_element
TEST(onnx2_stream, NextPackedElement) {
  std::vector<uint8_t> data = {// un float: 3.14
                               0xC3, 0xF5, 0x48, 0x40,
                               // un int32: 42 (stock� en format binaire direct)
                               0x2A, 0x00, 0x00, 0x00};

  utils::StringStream stream(data.data(), data.size());

  // Tester next_packed_element pour float
  float f;
  stream.next_packed_element(f);
  EXPECT_NEAR(f, 3.14f, 0.0001f);

  // Tester next_packed_element pour int32
  int32_t i;
  stream.next_packed_element(i);
  EXPECT_EQ(i, 42);
}

// Tests pour les cas d'erreur
TEST(onnx2_stream, ErrorCases) {
  // Tester next_uint64 avec des donn�es insuffisantes
  std::vector<uint8_t> badData = {0x80, 0x80, 0x80}; // Varint invalide (continue ind�finiment)
  utils::StringStream badStream(badData.data(), badData.size());

  EXPECT_THROW(badStream.next_uint64(), std::runtime_error);

  // Tester can_read avec une taille trop grande
  std::vector<uint8_t> smallData = {0x01, 0x02};
  utils::StringStream smallStream(smallData.data(), smallData.size());

  EXPECT_THROW(smallStream.can_read(3, "Test message"), std::runtime_error);
}

TEST(onnx2_proto, StringStringEntryProto_Basic) {
  StringStringEntryProto entry;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(entry.key().empty());
  EXPECT_TRUE(entry.value().empty());
  EXPECT_FALSE(entry.has_key());
  EXPECT_FALSE(entry.has_value());

  // Test des setters
  entry.set_key("test_key");
  entry.set_value("test_value");

  // Test des getters
  EXPECT_EQ(entry.key(), "test_key");
  EXPECT_EQ(entry.value(), "test_value");
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  // Test des ordres
  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx2_proto, StringStringEntryProto_Serialization) {
  StringStringEntryProto entry;
  entry.set_key("test_key");
  entry.set_value("test_value");

  // S�rialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  // V�rification
  EXPECT_EQ(entry2.key(), "test_key");
  EXPECT_EQ(entry2.value(), "test_value");
}

// Tests pour IntIntListEntryProto
TEST(onnx2_proto, IntIntListEntryProto_Basic) {
  IntIntListEntryProto entry;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(entry.key(), 0);
  EXPECT_EQ(entry.value().size(), 0);
  EXPECT_FALSE(entry.has_value());

  // Test des setters
  entry.set_key(42);
  entry.value().values.push_back(1);
  entry.value().values.push_back(2);
  entry.value().values.push_back(3);

  // Test des getters
  EXPECT_EQ(entry.key(), 42);
  EXPECT_EQ(entry.value().size(), 3);
  EXPECT_EQ(entry.value()[0], 1);
  EXPECT_EQ(entry.value()[1], 2);
  EXPECT_EQ(entry.value()[2], 3);
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  // Test des ordres
  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx2_proto, IntIntListEntryProto_Serialization) {
  IntIntListEntryProto entry;
  entry.set_key(42);
  entry.value().values.push_back(1);
  entry.value().values.push_back(2);
  entry.value().values.push_back(3);

  // S�rialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  IntIntListEntryProto entry2;
  entry2.ParseFromString(serialized);

  // V�rification
  EXPECT_EQ(entry2.key(), 42);
  EXPECT_EQ(entry2.value().size(), 3);
  EXPECT_EQ(entry2.value()[0], 1);
  EXPECT_EQ(entry2.value()[1], 2);
  EXPECT_EQ(entry2.value()[2], 3);
}

// Tests pour TensorAnnotation
TEST(onnx2_proto, TensorAnnotation_Basic) {
  TensorAnnotation annotation;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(annotation.tensor_name().empty());
  EXPECT_EQ(annotation.quant_parameter_tensor_names().size(), 0);

  // Test des setters
  annotation.set_tensor_name("my_tensor");
  StringStringEntryProto &entry = annotation.add_quant_parameter_tensor_names();
  entry.set_key("scale");
  entry.set_value("scale_tensor");

  // Test des getters
  EXPECT_EQ(annotation.tensor_name(), "my_tensor");
  EXPECT_EQ(annotation.quant_parameter_tensor_names().size(), 1);
  EXPECT_EQ(annotation.quant_parameter_tensor_names()[0].key(), "scale");
  EXPECT_EQ(annotation.quant_parameter_tensor_names()[0].value(), "scale_tensor");
}

// Tests pour DeviceConfigurationProto
TEST(onnx2_proto, DeviceConfigurationProto_Basic) {
  DeviceConfigurationProto config;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(config.name().empty());
  EXPECT_EQ(config.num_devices(), 0);
  EXPECT_EQ(config.device().size(), 0);

  // Test des setters
  config.set_name("CPU");
  config.set_num_devices(2);
  config.add_device() = "device0";
  config.add_device() = "device1";

  // Test des getters
  EXPECT_EQ(config.name(), "CPU");
  EXPECT_EQ(config.num_devices(), 2);
  EXPECT_EQ(config.device().size(), 2);
  EXPECT_EQ(config.device()[0], "device0");
  EXPECT_EQ(config.device()[1], "device1");
}

// Tests pour SimpleShardedDimProto
TEST(onnx2_proto, SimpleShardedDimProto_Basic) {
  SimpleShardedDimProto dim;

  // Test des propri�t�s par d�faut
  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.dim_param().empty());
  EXPECT_EQ(dim.num_shards(), 0);

  // Test des setters
  dim.set_dim_value(100);
  dim.set_dim_param("batch");
  dim.set_num_shards(4);

  // Test des getters
  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.dim_value(), 100);
  EXPECT_EQ(dim.dim_param(), "batch");
  EXPECT_EQ(dim.num_shards(), 4);
}

// Tests pour ShardedDimProto
TEST(onnx2_proto, ShardedDimProto_Basic) {
  ShardedDimProto dim;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(dim.axis(), 0);
  EXPECT_EQ(dim.simple_sharding().size(), 0);

  // Test des setters
  dim.set_axis(1);
  SimpleShardedDimProto &simple_dim = dim.add_simple_sharding();
  simple_dim.set_dim_value(100);
  simple_dim.set_num_shards(4);

  // Test des getters
  EXPECT_EQ(dim.axis(), 1);
  EXPECT_EQ(dim.simple_sharding().size(), 1);
  EXPECT_EQ(dim.simple_sharding()[0].dim_value(), 100);
  EXPECT_EQ(dim.simple_sharding()[0].num_shards(), 4);
}

// Tests pour ShardingSpecProto
TEST(onnx2_proto, ShardingSpecProto_Basic) {
  ShardingSpecProto spec;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(spec.tensor_name().empty());
  EXPECT_EQ(spec.device().size(), 0);
  EXPECT_EQ(spec.index_to_device_group_map().size(), 0);
  EXPECT_EQ(spec.sharded_dim().size(), 0);

  // Test des setters
  spec.set_tensor_name("my_tensor");

  spec.device().values.push_back(0);
  spec.device().values.push_back(1);

  IntIntListEntryProto &map_entry = spec.add_index_to_device_group_map();
  map_entry.set_key(0);
  map_entry.value().values.push_back(0);

  ShardedDimProto &dim = spec.add_sharded_dim();
  dim.set_axis(0);

  // Test des getters
  EXPECT_EQ(spec.tensor_name(), "my_tensor");
  EXPECT_EQ(spec.device().size(), 2);
  EXPECT_EQ(spec.device()[0], 0);
  EXPECT_EQ(spec.device()[1], 1);
  EXPECT_EQ(spec.index_to_device_group_map().size(), 1);
  EXPECT_EQ(spec.index_to_device_group_map()[0].key(), 0);
  EXPECT_EQ(spec.sharded_dim().size(), 1);
  EXPECT_EQ(spec.sharded_dim()[0].axis(), 0);
}

// Tests pour NodeDeviceConfigurationProto
TEST(onnx2_proto, NodeDeviceConfigurationProto_Basic) {
  NodeDeviceConfigurationProto config;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(config.configuration_id().empty());
  EXPECT_EQ(config.sharding_spec().size(), 0);
  EXPECT_FALSE(config.has_pipeline_stage());

  // Test des setters
  config.set_configuration_id("config1");
  config.add_sharding_spec();
  config.set_pipeline_stage(2);

  // Test des getters
  EXPECT_EQ(config.configuration_id(), "config1");
  EXPECT_EQ(config.sharding_spec().size(), 1);
  EXPECT_TRUE(config.has_pipeline_stage());
  EXPECT_EQ(config.pipeline_stage(), 2);
}

// Tests pour OperatorSetIdProto
TEST(onnx2_proto, OperatorSetIdProto_Basic) {
  OperatorSetIdProto op_set;

  // Test des propri�t�s par d�faut
  EXPECT_TRUE(op_set.domain().empty());
  EXPECT_EQ(op_set.version(), 0);

  // Test des setters
  op_set.set_domain("ai.onnx");
  op_set.set_version(12);

  // Test des getters
  EXPECT_EQ(op_set.domain(), "ai.onnx");
  EXPECT_EQ(op_set.version(), 12);
}

// Tests pour TensorShapeProto
TEST(onnx2_proto, TensorShapeProto_Basic) {
  TensorShapeProto shape;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(shape.dim().size(), 0);

  // Test d'ajout de dimensions
  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.dim().add();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  // Test des getters
  EXPECT_EQ(shape.dim().size(), 2);
  EXPECT_TRUE(shape.dim()[0].has_dim_value());
  EXPECT_EQ(shape.dim()[0].dim_value(), 5);
  EXPECT_FALSE(shape.dim()[0].has_dim_param());

  EXPECT_FALSE(shape.dim()[1].has_dim_value());
  EXPECT_EQ(shape.dim()[1].dim_param(), "N");
  EXPECT_EQ(shape.dim()[1].denotation(), "batch");
}

TEST(onnx2_proto, TensorShapeProto_Dimension) {
  TensorShapeProto::Dimension dim;

  // Test des propri�t�s par d�faut
  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.dim_param().empty());
  EXPECT_TRUE(dim.denotation().empty());

  // Test des setters pour dim_value
  dim.set_dim_value(10);
  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.dim_value(), 10);

  // Test des setters pour dim_param
  dim.set_dim_param("batch_size");
  EXPECT_EQ(dim.dim_param(), "batch_size");

  // Test des setters pour denotation
  dim.set_denotation("batch");
  EXPECT_EQ(dim.denotation(), "batch");
}

// Tests pour TensorProto
TEST(onnx2_proto, TensorProto_Basic) {
  TensorProto tensor;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::UNDEFINED);
  EXPECT_EQ(tensor.dims().size(), 0);
  EXPECT_TRUE(tensor.name().empty());

  // Test des setters
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(3);
  tensor.set_name("my_tensor");

  // Ajout de donn�es
  tensor.float_data().values.push_back(1.0f);
  tensor.float_data().values.push_back(2.0f);
  tensor.float_data().values.push_back(3.0f);
  tensor.float_data().values.push_back(4.0f);
  tensor.float_data().values.push_back(5.0f);
  tensor.float_data().values.push_back(6.0f);

  // Test des getters
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 2);
  EXPECT_EQ(tensor.dims()[1], 3);
  EXPECT_EQ(tensor.name(), "my_tensor");
  EXPECT_EQ(tensor.float_data().size(), 6);
  EXPECT_EQ(tensor.float_data()[0], 1.0f);
  EXPECT_EQ(tensor.float_data()[5], 6.0f);
}

TEST(onnx2_proto, TensorProto_DataTypes) {
  TensorProto tensor;

  // Test avec float_data
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.float_data().values.push_back(1.0f);
  tensor.float_data().values.push_back(2.0f);
  EXPECT_EQ(tensor.float_data().size(), 2);
  EXPECT_EQ(tensor.float_data()[0], 1.0f);
  EXPECT_EQ(tensor.float_data()[1], 2.0f);

  // Test avec int32_data
  tensor.set_data_type(TensorProto::DataType::INT32);
  tensor.int32_data().values.push_back(10);
  tensor.int32_data().values.push_back(20);
  EXPECT_EQ(tensor.int32_data().size(), 2);
  EXPECT_EQ(tensor.int32_data()[0], 10);
  EXPECT_EQ(tensor.int32_data()[1], 20);

  // Test avec string_data
  tensor.set_data_type(TensorProto::DataType::STRING);
  tensor.add_string_data() = "hello";
  tensor.add_string_data() = "world";
  EXPECT_EQ(tensor.string_data().size(), 2);
  EXPECT_EQ(tensor.string_data()[0], "hello");
  EXPECT_EQ(tensor.string_data()[1], "world");

  // Test avec int64_data
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.int64_data().values.push_back(100);
  tensor.int64_data().values.push_back(200);
  EXPECT_EQ(tensor.int64_data().size(), 2);
  EXPECT_EQ(tensor.int64_data()[0], 100);
  EXPECT_EQ(tensor.int64_data()[1], 200);

  // Test avec double_data
  tensor.set_data_type(TensorProto::DataType::DOUBLE);
  tensor.double_data().values.push_back(1.5);
  tensor.double_data().values.push_back(2.5);
  EXPECT_EQ(tensor.double_data().size(), 2);
  EXPECT_EQ(tensor.double_data()[0], 1.5);
  EXPECT_EQ(tensor.double_data()[1], 2.5);

  // Test avec uint64_data
  tensor.set_data_type(TensorProto::DataType::UINT64);
  tensor.uint64_data().values.push_back(1000);
  tensor.uint64_data().values.push_back(2000);
  EXPECT_EQ(tensor.uint64_data().size(), 2);
  EXPECT_EQ(tensor.uint64_data()[0], 1000);
  EXPECT_EQ(tensor.uint64_data()[1], 2000);
}

TEST(onnx2_proto, TensorProto_Segment) {
  TensorProto tensor;

  // Test des propri�t�s par d�faut du segment
  EXPECT_EQ(tensor.segment().begin(), 0);
  EXPECT_EQ(tensor.segment().end(), 0);

  // Test des setters
  tensor.segment().set_begin(5);
  tensor.segment().set_end(10);

  // Test des getters
  EXPECT_EQ(tensor.segment().begin(), 5);
  EXPECT_EQ(tensor.segment().end(), 10);
}

TEST(onnx2_proto, TensorProto_RawData) {
  TensorProto tensor;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(tensor.raw_data().size(), 0);

  // Pr�paration de donn�es
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Copie dans raw_data
  tensor.raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.raw_data().data(), data.data(), data.size() * sizeof(float));

  // Test de la taille
  EXPECT_EQ(tensor.raw_data().size(), data.size() * sizeof(float));

  // V�rification des donn�es
  const float *raw_data_ptr = reinterpret_cast<const float *>(tensor.raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

TEST(onnx2_proto, TensorProto_Serialization) {
  TensorProto tensor1;
  tensor1.set_name("test_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.dims().values.push_back(2);
  tensor1.dims().values.push_back(2);
  tensor1.float_data().values.push_back(1.0f);
  tensor1.float_data().values.push_back(2.0f);
  tensor1.float_data().values.push_back(3.0f);
  tensor1.float_data().values.push_back(4.0f);

  // S�rialisation
  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // V�rification
  EXPECT_EQ(tensor2.name(), "test_tensor");
  EXPECT_EQ(tensor2.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.dims().size(), 2);
  EXPECT_EQ(tensor2.dims()[0], 2);
  EXPECT_EQ(tensor2.dims()[1], 2);
  EXPECT_EQ(tensor2.float_data().size(), 4);
  EXPECT_EQ(tensor2.float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.float_data()[3], 4.0f);
}

// Tests pour SparseTensorProto
TEST(onnx2_proto, SparseTensorProto_Basic) {
  SparseTensorProto sparse;

  // Test des propri�t�s par d�faut
  EXPECT_EQ(sparse.dims().size(), 0);

  // Configuration d'un tenseur sparse
  sparse.dims().values.push_back(3);
  sparse.dims().values.push_back(4);

  // Valeurs: [5, 6]
  sparse.values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.values().float_data().values.push_back(5.0f);
  sparse.values().float_data().values.push_back(6.0f);

  // Indices: [(0,2), (1,3)]
  sparse.indices().set_data_type(TensorProto::DataType::INT64);
  sparse.indices().dims().values.push_back(2); // Nombre d'�l�ments non nuls
  sparse.indices().dims().values.push_back(2); // Nombre de dimensions
  sparse.indices().int64_data().values.push_back(0);
  sparse.indices().int64_data().values.push_back(2);
  sparse.indices().int64_data().values.push_back(1);
  sparse.indices().int64_data().values.push_back(3);

  // Test des getters
  EXPECT_EQ(sparse.dims().size(), 2);
  EXPECT_EQ(sparse.dims()[0], 3);
  EXPECT_EQ(sparse.dims()[1], 4);

  EXPECT_EQ(sparse.values().data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.values().float_data().size(), 2);
  EXPECT_EQ(sparse.values().float_data()[0], 5.0f);
  EXPECT_EQ(sparse.values().float_data()[1], 6.0f);

  EXPECT_EQ(sparse.indices().data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.indices().int64_data().size(), 4);
  EXPECT_EQ(sparse.indices().int64_data()[0], 0);
  EXPECT_EQ(sparse.indices().int64_data()[1], 2);
  EXPECT_EQ(sparse.indices().int64_data()[2], 1);
  EXPECT_EQ(sparse.indices().int64_data()[3], 3);
}

// Tests pour TypeProto
TEST(onnx2_proto, TypeProto_Tensor) {
  TypeProto type;

  // Test des propri�t�s par d�faut
  EXPECT_FALSE(type.has_tensor_type());

  // Configuration du tensor_type
  type.add_tensor_type().set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_FALSE(type.tensor_type().has_shape());
  TensorShapeProto &shape = type.tensor_type().add_shape();
  EXPECT_TRUE(type.tensor_type().has_shape());
  TensorShapeProto::Dimension &dim = shape.add_dim();
  dim.set_dim_value(3);

  // Test des getters
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), 1);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim().size(), 1);
  EXPECT_EQ(type.tensor_type().shape().dim()[0].dim_value(), 3);
}

TEST(onnx2_proto, CreateTensorProto) {
  // Test de cr�ation d'un TensorProto
  TensorProto tensor;
  tensor.set_name("test_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(3);

  // Ajout de donn�es
  for (int i = 0; i < 6; ++i) {
    tensor.float_data().values.push_back(static_cast<float>(i + 1));
  }

  // V�rification des propri�t�s
  EXPECT_EQ(tensor.name(), "test_tensor");
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 2);
  EXPECT_EQ(tensor.dims()[1], 3);
  EXPECT_EQ(tensor.float_data().size(), 6);
}

// Test pour v�rifier la s�rialisation/d�s�rialisation d'un TensorProto
TEST(onnx2_proto, SerializeDeserializeTensorProto) {
  // Cr�ation d'un TensorProto
  TensorProto tensor1;
  tensor1.set_name("serialized_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.dims().values.push_back(2);
  tensor1.dims().values.push_back(2);
  tensor1.float_data().values.push_back(1.0f);
  tensor1.float_data().values.push_back(2.0f);
  tensor1.float_data().values.push_back(3.0f);
  tensor1.float_data().values.push_back(4.0f);

  // S�rialisation
  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // V�rification que les deux tenseurs sont identiques
  EXPECT_EQ(tensor2.name(), "serialized_tensor");
  EXPECT_EQ(tensor2.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.dims().size(), 2);
  EXPECT_EQ(tensor2.dims()[0], 2);
  EXPECT_EQ(tensor2.dims()[1], 2);
  EXPECT_EQ(tensor2.float_data().size(), 4);
  EXPECT_EQ(tensor2.float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.float_data()[3], 4.0f);
}

// Test pour v�rifier la cr�ation et la manipulation d'un TypeProto
TEST(onnx2_proto, TypeProtoOperations) {
  // Cr�ation d'un TypeProto
  TypeProto type;

  // Configuration du tensor_type
  type.add_tensor_type().set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());

  // Ajout d'une forme
  TensorShapeProto &shape = type.tensor_type().add_shape();

  // Ajout de dimensions
  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(3);

  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("batch_size");

  // V�rification
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), 1);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim().size(), 2);
  EXPECT_EQ(type.tensor_type().shape().dim()[0].dim_value(), 3);
  EXPECT_EQ(type.tensor_type().shape().dim()[1].dim_param(), "batch_size");
}

// Test pour v�rifier la cr�ation et la s�rialisation d'un StringStringEntryProto
TEST(onnx2_proto, StringStringEntryProtoOperations) {
  // Cr�ation d'un StringStringEntryProto
  StringStringEntryProto entry;
  entry.set_key("metadata_key");
  entry.set_value("metadata_value");

  // V�rification des propri�t�s
  EXPECT_EQ(entry.key(), "metadata_key");
  EXPECT_EQ(entry.value(), "metadata_value");

  // S�rialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  // V�rification
  EXPECT_EQ(entry2.key(), "metadata_key");
  EXPECT_EQ(entry2.value(), "metadata_value");
}

// Test pour v�rifier la cr�ation et la manipulation d'un TensorProto avec raw_data
TEST(onnx2_proto, TensorProtoWithRawData) {
  // Cr�ation d'un TensorProto
  TensorProto tensor;
  tensor.set_name("raw_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(2);

  // Pr�paration des donn�es
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Copie dans raw_data
  tensor.raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.raw_data().data(), data.data(), data.size() * sizeof(float));

  // V�rification
  EXPECT_EQ(tensor.name(), "raw_data_tensor");
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 2);
  EXPECT_EQ(tensor.dims()[1], 2);
  EXPECT_EQ(tensor.raw_data().size(), data.size() * sizeof(float));

  // V�rification du contenu de raw_data
  const float *raw_data_ptr = reinterpret_cast<const float *>(tensor.raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

// Test pour v�rifier la cr�ation et manipulation d'un SparseTensorProto
TEST(onnx2_proto, SparseTensorProtoOperations) {
  // Cr�ation d'un SparseTensorProto
  SparseTensorProto sparse;

  // Configuration des dimensions
  sparse.dims().values.push_back(3);
  sparse.dims().values.push_back(4);

  // Configuration des valeurs
  sparse.values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.values().float_data().values.push_back(5.0f);
  sparse.values().float_data().values.push_back(6.0f);

  // Configuration des indices
  sparse.indices().set_data_type(TensorProto::DataType::INT64);
  sparse.indices().dims().values.push_back(2); // Nombre d'�l�ments non nuls
  sparse.indices().dims().values.push_back(2); // Nombre de dimensions
  sparse.indices().int64_data().values.push_back(0);
  sparse.indices().int64_data().values.push_back(2);
  sparse.indices().int64_data().values.push_back(1);
  sparse.indices().int64_data().values.push_back(3);

  // V�rification
  EXPECT_EQ(sparse.dims().size(), 2);
  EXPECT_EQ(sparse.dims()[0], 3);
  EXPECT_EQ(sparse.dims()[1], 4);

  EXPECT_EQ(sparse.values().data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.values().float_data().size(), 2);
  EXPECT_EQ(sparse.values().float_data()[0], 5.0f);
  EXPECT_EQ(sparse.values().float_data()[1], 6.0f);

  EXPECT_EQ(sparse.indices().data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.indices().int64_data().size(), 4);
  EXPECT_EQ(sparse.indices().int64_data()[0], 0);
  EXPECT_EQ(sparse.indices().int64_data()[1], 2);
  EXPECT_EQ(sparse.indices().int64_data()[2], 1);
  EXPECT_EQ(sparse.indices().int64_data()[3], 3);

  // S�rialisation
  std::string serialized;
  sparse.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  SparseTensorProto sparse2;
  sparse2.ParseFromString(serialized);

  // V�rification apr�s d�s�rialisation
  EXPECT_EQ(sparse2.dims().size(), 2);
  EXPECT_EQ(sparse2.values().float_data().size(), 2);
  EXPECT_EQ(sparse2.indices().int64_data().size(), 4);
}

// Test pour v�rifier les manipulations d'un TensorShapeProto
TEST(onnx2_proto, TensorShapeProtoOperations) {
  // Cr�ation d'un TensorShapeProto
  TensorShapeProto shape;

  // Ajout de dimensions
  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  // V�rification
  EXPECT_EQ(shape.dim().size(), 2);
  EXPECT_TRUE(shape.dim()[0].has_dim_value());
  EXPECT_EQ(shape.dim()[0].dim_value(), 5);
  EXPECT_FALSE(shape.dim()[0].has_dim_param());

  EXPECT_FALSE(shape.dim()[1].has_dim_value());
  EXPECT_EQ(shape.dim()[1].dim_param(), "N");
  EXPECT_EQ(shape.dim()[1].denotation(), "batch");

  // S�rialisation
  std::string serialized;
  shape.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // D�s�rialisation
  TensorShapeProto shape2;
  shape2.ParseFromString(serialized);

  // V�rification apr�s d�s�rialisation
  EXPECT_EQ(shape2.dim().size(), 2);
  EXPECT_EQ(shape2.dim()[0].dim_value(), 5);
  EXPECT_EQ(shape2.dim()[1].dim_param(), "N");
  EXPECT_EQ(shape2.dim()[1].denotation(), "batch");
}

// Test pour v�rifier le comportement avec diff�rents types de donn�es dans TensorProto
TEST(onnx2_proto, TensorProtoDataTypes) {
  // Test avec float_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.float_data().values.push_back(1.0f);
    tensor.float_data().values.push_back(2.0f);
    EXPECT_EQ(tensor.float_data().size(), 2);
    EXPECT_EQ(tensor.float_data()[0], 1.0f);
    EXPECT_EQ(tensor.float_data()[1], 2.0f);
  }

  // Test avec int32_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT32);
    tensor.int32_data().values.push_back(10);
    tensor.int32_data().values.push_back(20);
    EXPECT_EQ(tensor.int32_data().size(), 2);
    EXPECT_EQ(tensor.int32_data()[0], 10);
    EXPECT_EQ(tensor.int32_data()[1], 20);
  }

  // Test avec string_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::STRING);
    tensor.add_string_data() = "hello";
    tensor.add_string_data() = "world";
    EXPECT_EQ(tensor.string_data().size(), 2);
    EXPECT_EQ(tensor.string_data()[0], "hello");
    EXPECT_EQ(tensor.string_data()[1], "world");
  }

  // Test avec int64_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT64);
    tensor.int64_data().values.push_back(100);
    tensor.int64_data().values.push_back(200);
    EXPECT_EQ(tensor.int64_data().size(), 2);
    EXPECT_EQ(tensor.int64_data()[0], 100);
    EXPECT_EQ(tensor.int64_data()[1], 200);
  }

  // Test avec double_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::DOUBLE);
    tensor.double_data().values.push_back(1.5);
    tensor.double_data().values.push_back(2.5);
    EXPECT_EQ(tensor.double_data().size(), 2);
    EXPECT_EQ(tensor.double_data()[0], 1.5);
    EXPECT_EQ(tensor.double_data()[1], 2.5);
  }

  // Test avec uint64_data
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT64);
    tensor.uint64_data().values.push_back(1000);
    tensor.uint64_data().values.push_back(2000);
    EXPECT_EQ(tensor.uint64_data().size(), 2);
    EXPECT_EQ(tensor.uint64_data()[0], 1000);
    EXPECT_EQ(tensor.uint64_data()[1], 2000);
  }
}

static TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
  case TensorProto_DataType::TensorProto_DataType_FLOAT:
    t.add_float_data((float)value);
    break;
  case TensorProto_DataType::TensorProto_DataType_DOUBLE:
    t.add_double_data(value);
    break;
  default:
    assert(false);
  }
  return t;
}

TEST(onnx2onnx, DataType) {
  TensorProto proto = ToTensor(4.5, TensorProto_DataType::TensorProto_DataType_FLOAT);
  EXPECT_EQ(proto.float_data().size(), 1);
  EXPECT_EQ(proto.float_data()[0], 4.5);
  EXPECT_EQ(proto.data_type(), TensorProto_DataType::TensorProto_DataType_FLOAT);
}

TEST(serialize_to_string, StringStringEntryProto) {
  onnx2::StringStringEntryProto proto;
  proto.set_key("test_key");
  proto.set_value("test_value");
  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_EQ(1, result.size());
  std::string serialized = result[0];
  EXPECT_TRUE(serialized.find("test_key") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_value") != std::string::npos);
}

TEST(serialize_to_string, IntIntListEntryProto) {
  onnx2::IntIntListEntryProto proto;
  proto.set_key(42);
  proto.value().values.push_back(1);
  proto.value().values.push_back(2);
  proto.value().values.push_back(3);
  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_EQ(5, result.size());
  std::string serialized = utils::join_string(result, "\n");
  EXPECT_TRUE(serialized.find("42") != std::string::npos);
  EXPECT_TRUE(serialized.find("1") != std::string::npos);
  EXPECT_TRUE(serialized.find("2") != std::string::npos);
  EXPECT_TRUE(serialized.find("3") != std::string::npos);
}

TEST(serialize_to_string, TensorAnnotation) {
  onnx2::TensorAnnotation proto;
  proto.set_tensor_name("my_tensor");
  auto &entry = proto.add_quant_parameter_tensor_names();
  entry.set_key("scale");
  entry.set_value("scale_tensor");
  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_EQ(6, result.size());
  std::string serialized = utils::join_string(result, "\n");
  EXPECT_TRUE(serialized.find("my_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale_tensor") != std::string::npos);
}

TEST(serialize_to_string, DeviceConfigurationProto) {
  DeviceConfigurationProto config;
  config.set_name("test_device_config");
  config.set_num_devices(3);
  config.add_device() = "device1";
  config.add_device() = "device2";
  config.add_device() = "device3";

  std::vector<std::string> result = config.SerializeToVectorString();

  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundNumDevices = false;
  bool foundDevices = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("name:") != std::string::npos &&
      item.find("test_device_config") != std::string::npos) {
    foundName = true;
  }
  if (item.find("num_devices:") != std::string::npos && item.find("3") != std::string::npos) {
    foundNumDevices = true;
  }
  if (item.find("device:") != std::string::npos && item.find("device1") != std::string::npos &&
      item.find("device2") != std::string::npos && item.find("device3") != std::string::npos) {
    foundDevices = true;
  }

  EXPECT_TRUE(foundName) << "Le nom du dispositif n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundNumDevices)
      << "Le nombre de dispositifs n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundDevices) << "La liste des dispositifs n'a pas �t� trouv�e dans le r�sultat";
}

TEST(serialize_to_string, SimpleShardedDimProto) {
  onnx2::SimpleShardedDimProto proto;
  proto.set_dim_value(100);
  proto.set_dim_param("batch_size");
  proto.set_num_shards(4);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundDimValue = false;
  bool foundDimParam = false;
  bool foundNumShards = false;

  for (const auto &item : result) {
    if (item.find("dim_value:") != std::string::npos && item.find("100") != std::string::npos) {
      foundDimValue = true;
    }
    if (item.find("dim_param:") != std::string::npos &&
        item.find("batch_size") != std::string::npos) {
      foundDimParam = true;
    }
    if (item.find("num_shards:") != std::string::npos && item.find("4") != std::string::npos) {
      foundNumShards = true;
    }
  }

  EXPECT_TRUE(foundDimValue) << "La valeur de dimension n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundDimParam) << "Le param�tre de dimension n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundNumShards) << "Le nombre de shards n'a pas �t� trouv� dans le r�sultat";
}

TEST(serialize_to_string, ShardedDimProto) {
  onnx2::ShardedDimProto proto;
  proto.set_axis(2);

  auto &simple_dim1 = proto.add_simple_sharding();
  simple_dim1.set_dim_value(100);
  simple_dim1.set_num_shards(4);

  auto &simple_dim2 = proto.add_simple_sharding();
  simple_dim2.set_dim_param("height");
  simple_dim2.set_num_shards(2);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundAxis = false;
  bool foundSimpleSharding = false;

  for (const auto &item : result) {
    if (item.find("axis:") != std::string::npos && item.find("2") != std::string::npos) {
      foundAxis = true;
    }
    if (item.find("simple_sharding") != std::string::npos) {
      foundSimpleSharding = true;
    }
  }

  EXPECT_TRUE(foundAxis) << "L'axe n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundSimpleSharding)
      << "Les simple_sharding n'ont pas �t� trouv�s dans le r�sultat";
}

TEST(serialize_to_string, ShardingSpecProto) {
  onnx2::ShardingSpecProto proto;
  proto.set_tensor_name("sharded_tensor");

  // Ajouter des dispositifs
  proto.device().values.push_back(0);
  proto.device().values.push_back(1);
  proto.device().values.push_back(2);

  // Ajouter une entr�e de mapping
  auto &map_entry = proto.add_index_to_device_group_map();
  map_entry.set_key(0);
  map_entry.value().values.push_back(0);
  map_entry.value().values.push_back(1);

  // Ajouter une dimension shard�e
  auto &dim = proto.add_sharded_dim();
  dim.set_axis(1);
  auto &simple_dim = dim.add_simple_sharding();
  simple_dim.set_dim_value(64);
  simple_dim.set_num_shards(4);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundTensorName = false;
  bool foundDevice = false;
  bool foundMapping = false;
  bool foundShardedDim = false;

  for (const auto &item : result) {
    if (item.find("tensor_name:") != std::string::npos &&
        item.find("sharded_tensor") != std::string::npos) {
      foundTensorName = true;
    }
    if (item.find("device:") != std::string::npos) {
      foundDevice = true;
    }
    if (item.find("index_to_device_group_map") != std::string::npos) {
      foundMapping = true;
    }
    if (item.find("sharded_dim") != std::string::npos) {
      foundShardedDim = true;
    }
  }

  EXPECT_TRUE(foundTensorName) << "Le nom du tenseur n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundDevice) << "Les dispositifs n'ont pas �t� trouv�s dans le r�sultat";
  EXPECT_TRUE(foundMapping) << "Le mapping n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundShardedDim) << "La dimension shard�e n'a pas �t� trouv�e dans le r�sultat";
}

TEST(serialize_to_string, NodeDeviceConfigurationProto) {
  onnx2::NodeDeviceConfigurationProto proto;
  proto.set_configuration_id("node_config_1");
  proto.set_pipeline_stage(3);

  // Ajouter une sp�cification de sharding
  auto &spec = proto.add_sharding_spec();
  spec.set_tensor_name("input_tensor");
  spec.device().values.push_back(0);
  spec.device().values.push_back(1);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundConfigId = false;
  bool foundPipelineStage = false;
  bool foundShardingSpec = false;

  for (const auto &item : result) {
    if (item.find("configuration_id:") != std::string::npos &&
        item.find("node_config_1") != std::string::npos) {
      foundConfigId = true;
    }
    if (item.find("pipeline_stage:") != std::string::npos &&
        item.find("3") != std::string::npos) {
      foundPipelineStage = true;
    }
    if (item.find("sharding_spec") != std::string::npos) {
      foundShardingSpec = true;
    }
  }

  EXPECT_TRUE(foundConfigId) << "L'ID de configuration n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundPipelineStage) << "L'�tape de pipeline n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundShardingSpec)
      << "La sp�cification de sharding n'a pas �t� trouv�e dans le r�sultat";
}

TEST(serialize_to_string, OperatorSetIdProto) {
  onnx2::OperatorSetIdProto proto;
  proto.set_domain("ai.onnx");
  proto.set_version(15);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundDomain = false;
  bool foundVersion = false;

  for (const auto &item : result) {
    if (item.find("domain:") != std::string::npos &&
        item.find("ai.onnx") != std::string::npos) {
      foundDomain = true;
    }
    if (item.find("version:") != std::string::npos && item.find("15") != std::string::npos) {
      foundVersion = true;
    }
  }

  EXPECT_TRUE(foundDomain) << "Le domaine n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundVersion) << "La version n'a pas �t� trouv�e dans le r�sultat";
}

TEST(serialize_to_string, TensorShapeProto) {
  onnx2::TensorShapeProto proto;

  // Ajouter des dimensions
  auto &dim1 = proto.add_dim();
  dim1.set_dim_value(64);

  auto &dim2 = proto.add_dim();
  dim2.set_dim_param("batch");
  dim2.set_denotation("N");

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundDim1 = false;
  bool foundDim2 = false;
  bool foundDenotation = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("dim") != std::string::npos &&
      item.find("dim_value: 64") != std::string::npos) {
    foundDim1 = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDim2 = true;
  }
  if (item.find("denotation: \"N\"") != std::string::npos) {
    foundDenotation = true;
  }

  EXPECT_TRUE(foundDim1) << "La premi�re dimension n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundDim2) << "La deuxi�me dimension n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundDenotation) << "La d�notation n'a pas �t� trouv�e dans le r�sultat";
}

TEST(serialize_to_string, TensorProto) {
  onnx2::TensorProto proto;
  proto.set_name("test_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.dims().values.push_back(3);
  proto.dims().values.push_back(4);

  // Ajouter des donn�es
  for (int i = 0; i < 12; ++i) {
    proto.float_data().values.push_back(static_cast<float>(i * 0.5f));
  }

  // Ajouter des m�tadonn�es
  proto.doc_string() = "Un tenseur de test";

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDataType = false;
  bool foundDims = false;
  bool foundDocString = false;
  bool foundData = false;

  for (const auto &item : result) {
    if (item.find("name:") != std::string::npos &&
        item.find("test_tensor") != std::string::npos) {
      foundName = true;
    }
    if (item.find("data_type:") != std::string::npos &&
        item.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
            std::string::npos) {
      foundDataType = true;
    }
    if (item.find("dims:") != std::string::npos) {
      foundDims = true;
    }
    if (item.find("doc_string:") != std::string::npos &&
        item.find("Un tenseur de test") != std::string::npos) {
      foundDocString = true;
    }
    if (item.find("float_data") != std::string::npos) {
      foundData = true;
    }
  }

  EXPECT_TRUE(foundName) << "Le nom du tenseur n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundDataType) << "Le type de donn�es n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundDims) << "Les dimensions n'ont pas �t� trouv�es dans le r�sultat";
  EXPECT_TRUE(foundDocString)
      << "La cha�ne de documentation n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundData) << "Les donn�es float n'ont pas �t� trouv�es dans le r�sultat";
}

TEST(serialize_to_string, SparseTensorProto) {
  onnx2::SparseTensorProto proto;

  // Configuration des dimensions
  proto.dims().values.push_back(5);
  proto.dims().values.push_back(5);

  // Configuration des valeurs
  proto.values().set_name("values_tensor");
  proto.values().set_data_type(TensorProto::DataType::FLOAT);
  proto.values().float_data().values.push_back(1.5f);
  proto.values().float_data().values.push_back(2.5f);
  proto.values().float_data().values.push_back(3.5f);

  // Configuration des indices
  proto.indices().set_name("indices_tensor");
  proto.indices().set_data_type(TensorProto::DataType::INT64);
  proto.indices().dims().values.push_back(3); // 3 �l�ments non nuls
  proto.indices().dims().values.push_back(2); // coordonn�es 2D

  // Indices pour les 3 �l�ments non nuls : (0,1), (2,3), (4,2)
  proto.indices().int64_data().values.push_back(0);
  proto.indices().int64_data().values.push_back(1);
  proto.indices().int64_data().values.push_back(2);
  proto.indices().int64_data().values.push_back(3);
  proto.indices().int64_data().values.push_back(4);
  proto.indices().int64_data().values.push_back(2);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundDims = false;
  bool foundValues = false;
  bool foundIndices = false;

  for (const auto &item : result) {
    if (item.find("dims:") != std::string::npos && item.find("5") != std::string::npos) {
      foundDims = true;
    }
    if (item.find("values") != std::string::npos &&
        item.find("values_tensor") != std::string::npos) {
      foundValues = true;
    }
    if (item.find("indices") != std::string::npos &&
        item.find("indices_tensor") != std::string::npos) {
      foundIndices = true;
    }
  }

  EXPECT_TRUE(foundDims) << "Les dimensions n'ont pas �t� trouv�es dans le r�sultat";
  EXPECT_TRUE(foundValues) << "Les valeurs n'ont pas �t� trouv�es dans le r�sultat";
  EXPECT_TRUE(foundIndices) << "Les indices n'ont pas �t� trouv�s dans le r�sultat";
}

TEST(serialize_to_string, TypeProto) {
  onnx2::TypeProto proto;

  // Configuration du type tenseur
  proto.add_tensor_type().set_elem_type(1); // FLOAT

  // Configuration de la forme
  auto &shape = proto.tensor_type().add_shape();

  // Ajouter deux dimensions
  auto &dim1 = shape.add_dim();
  dim1.set_dim_value(10);

  auto &dim2 = shape.add_dim();
  dim2.set_dim_param("batch");
  dim2.set_denotation("N");

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundTensorType = false;
  bool foundElemType = false;
  bool foundShape = false;
  bool foundDimValue = false;
  bool foundDimParam = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("tensor_type") != std::string::npos) {
    foundTensorType = true;
  }
  if (item.find("elem_type: 1") != std::string::npos) {
    foundElemType = true;
  }
  if (item.find("shape") != std::string::npos) {
    foundShape = true;
  }
  if (item.find("dim_value: 10") != std::string::npos) {
    foundDimValue = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDimParam = true;
  }

  EXPECT_TRUE(foundTensorType) << "Le type tenseur n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundElemType) << "Le type d'�l�ment n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundShape) << "La forme n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundDimValue) << "La valeur de dimension n'a pas �t� trouv�e dans le r�sultat";
  EXPECT_TRUE(foundDimParam) << "Le param�tre de dimension n'a pas �t� trouv� dans le r�sultat";
}

TEST(serialize_to_string, TensorProto_WithRawData) {
  onnx2::TensorProto proto;
  proto.set_name("raw_data_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.dims().values.push_back(2);
  proto.dims().values.push_back(2);

  // Pr�paration des donn�es
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Copie dans raw_data
  proto.raw_data().resize(data.size() * sizeof(float));
  std::memcpy(proto.raw_data().data(), data.data(), data.size() * sizeof(float));

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDataType = false;
  bool foundRawData = false;

  for (const auto &item : result) {
    if (item.find("name:") != std::string::npos &&
        item.find("raw_data_tensor") != std::string::npos) {
      foundName = true;
    }
    if (item.find("data_type:") != std::string::npos &&
        item.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
            std::string::npos) {
      foundDataType = true;
    }
    if (item.find("raw_data:") != std::string::npos) {
      foundRawData = true;
    }
  }

  EXPECT_TRUE(foundName) << "Le nom du tenseur n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundDataType) << "Le type de donn�es n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundRawData) << "Les donn�es brutes n'ont pas �t� trouv�es dans le r�sultat";
}

TEST(serialize_to_string, TensorProto_WithSegment) {
  onnx2::TensorProto proto;
  proto.set_name("segmented_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);

  // Configuration du segment
  proto.segment().set_begin(5);
  proto.segment().set_end(10);

  std::vector<std::string> result = proto.SerializeToVectorString();
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundSegmentBegin = false;
  bool foundSegmentEnd = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("name:") != std::string::npos &&
      item.find("segmented_tensor") != std::string::npos) {
    foundName = true;
  }
  if (item.find("segment") != std::string::npos && item.find("begin: 5") != std::string::npos) {
    foundSegmentBegin = true;
  }
  if (item.find("segment") != std::string::npos && item.find("end: 10") != std::string::npos) {
    foundSegmentEnd = true;
  }

  EXPECT_TRUE(foundName) << "Le nom du tenseur n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundSegmentBegin) << "Le d�but du segment n'a pas �t� trouv� dans le r�sultat";
  EXPECT_TRUE(foundSegmentEnd) << "La fin du segment n'a pas �t� trouv�e dans le r�sultat";
}