#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
#include <gtest/gtest.h>

using namespace onnx2;

TEST(onnx2_string, RefString_Constructors) {
  // Test de constructeur avec copie
  utils::RefString original("test", 4);
  utils::RefString copied(original);
  EXPECT_EQ(copied.size(), 4);
  EXPECT_EQ(copied.data(), original.data());
  EXPECT_EQ(copied, original);

  // Test de constructeur avec pointeur et taille
  const char *text = "hello";
  utils::RefString rs(text, 5);
  EXPECT_EQ(rs.size(), 5);
  EXPECT_EQ(rs.data(), text);
}

TEST(onnx2_string, RefString_Assignment) {
  // Test d'assignation RefString à RefString
  utils::RefString a("abc", 3);
  utils::RefString b("xyz", 3);
  b = a;
  EXPECT_EQ(b.data(), a.data());
  EXPECT_EQ(b.size(), 3);

  // Test d'assignation String à RefString
  utils::String s("def", 3);
  utils::RefString c("123", 3);
  c = s;
  EXPECT_EQ(c.data(), s.data());
  EXPECT_EQ(c.size(), 3);
}

TEST(onnx2_string, RefString_Methods) {
  utils::RefString a("hello", 5);

  // Test size()
  EXPECT_EQ(a.size(), 5);

  // Test c_str() et data()
  EXPECT_EQ(a.c_str(), a.data());

  // Test empty()
  EXPECT_FALSE(a.empty());
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty.empty());

  // Test operator[]
  EXPECT_EQ(a[0], 'h');
  EXPECT_EQ(a[4], 'o');
}

TEST(onnx2_string, RefString_Equality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  std::string e("test");

  // Test RefString == RefString
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);

  // Test RefString == String
  EXPECT_TRUE(a == d);

  // Test RefString == std::string
  EXPECT_TRUE(a == e);

  // Test RefString == const char*
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");

  // Test avec chaîne vide
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

  // Test RefString != RefString
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  // Test RefString != String
  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);

  // Test RefString != std::string
  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);

  // Test RefString != const char*
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
  // Test constructeur par défaut
  utils::String defaultStr;
  EXPECT_EQ(defaultStr.size(), 0);
  EXPECT_EQ(defaultStr.data(), nullptr);
  EXPECT_TRUE(defaultStr.empty());

  // Test constructeur depuis RefString
  utils::RefString ref("test", 4);
  utils::String fromRef(ref);
  EXPECT_EQ(fromRef.size(), 4);
  EXPECT_NE(fromRef.data(), ref.data()); // Les données devraient être copiées
  EXPECT_EQ(fromRef, ref);

  // Test constructeur depuis char* et taille
  utils::String fromCharPtr("hello", 5);
  EXPECT_EQ(fromCharPtr.size(), 5);
  EXPECT_EQ(fromCharPtr, "hello");

  // Test constructeur avec string terminé par null
  utils::String withNull("abc\0", 4);
  EXPECT_EQ(withNull.size(), 3);
  EXPECT_EQ(withNull, "abc");

  // Test constructeur depuis std::string
  std::string stdStr = "world";
  utils::String fromStdStr(stdStr);
  EXPECT_EQ(fromStdStr.size(), 5);
  EXPECT_EQ(fromStdStr, stdStr);
}

TEST(onnx2_string, String_Assignment) {
  utils::String s;

  // Test d'assignation depuis const char*
  s = "abc";
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, "abc");

  // Test d'assignation depuis RefString
  utils::RefString ref("def", 3);
  s = ref;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, ref);

  // Test d'assignation depuis String
  utils::String other("xyz", 3);
  s = other;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data()); // Les données doivent être copiées

  // Test d'assignation depuis std::string
  std::string stdStr = "hello";
  s = stdStr;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s, stdStr);
}

TEST(onnx2_string, String_Methods) {
  utils::String s("hello", 5);

  // Test size()
  EXPECT_EQ(s.size(), 5);

  // Test data()
  EXPECT_NE(s.data(), nullptr);

  // Test empty()
  EXPECT_FALSE(s.empty());
  utils::String empty;
  EXPECT_TRUE(empty.empty());

  // Test operator[]
  EXPECT_EQ(s[0], 'h');
  EXPECT_EQ(s[4], 'o');

  // Test clear()
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

  // Test String == String
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);

  // Test String == RefString
  EXPECT_TRUE(a == d);

  // Test String == std::string
  EXPECT_TRUE(a == e);

  // Test String == const char*
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");

  // Test avec chaîne vide
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

  // Test String != String
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  // Test String != RefString
  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);

  // Test String != std::string
  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);

  // Test String != const char*
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
  // Test avec une chaîne vide
  utils::String empty("");
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);

  // Test avec une chaîne nullptr
  utils::String null(nullptr, 0);
  EXPECT_TRUE(null.empty());
  EXPECT_EQ(null.size(), 0);

  // Test avec une chaîne contenant des caractères nuls
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
  // Vérification de l'encodage/décodage ZigZag pour les entiers signés
  int64_t original_values[] = {0, -1, 1, -2, 2, INT64_MAX, INT64_MIN};

  for (auto val : original_values) {
    uint64_t encoded = utils::encodeZigZag64(val);
    int64_t decoded = utils::decodeZigZag64(encoded);
    EXPECT_EQ(decoded, val) << "ZigZag encoding/decoding failed for value: " << val;
  }
}

// Tests pour la classe FieldNumber
TEST(onnx2_stream, FieldNumber) {
  utils::FieldNumber fn;
  fn.field_number = 5;
  fn.wire_type = 2;

  std::string str = fn.string();
  EXPECT_FALSE(str.empty());
  EXPECT_NE(str.find("field_number=5"), std::string::npos);
  EXPECT_NE(str.find("wire_type=2"), std::string::npos);
}

// Tests pour StringStream
class onnx2_stream_2 : public ::testing::Test {
protected:
  void SetUp() override {
    // Préparation des données
    data = {// uint64_t encodé en varint: 150 (10010110 00000001 -> 10010110 10000000)
            0x96, 0x01,
            // int64_t encodé: 42
            0x2A,
            // int32_t encodé: 24
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
  // Sauter le premier uint64_t
  stream.next_uint64();

  int64_t value = stream.next_int64();
  EXPECT_EQ(value, 42);
}

TEST_F(onnx2_stream_2, NextInt32) {
  // Sauter les deux premiers nombres
  stream.next_uint64();
  stream.next_int64();

  int32_t value = stream.next_int32();
  EXPECT_EQ(value, 24);
}

TEST_F(onnx2_stream_2, NextFloat) {
  // Positionner le stream après les 3 premiers entiers
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();

  float value = stream.next_float();
  EXPECT_NEAR(value, 3.14f, 0.0001f);
}

TEST_F(onnx2_stream_2, NextField) {
  // Positionner le stream après les nombres
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
  // Positionner le stream juste avant la chaîne
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
  // Lire les 2 premiers octets
  const uint8_t *bytes = stream.read_bytes(2);
  EXPECT_EQ(bytes[0], 0x96);
  EXPECT_EQ(bytes[1], 0x01);
}

TEST_F(onnx2_stream_2, CanRead) {
  // Test positif
  stream.can_read(data.size(), "Test message");

  // Consommer quelques octets
  stream.read_bytes(10);

  // Test avec une taille limite
  stream.can_read(data.size() - 10, "Test message");

  // Test avec une taille trop grande (doit lever une exception)
  EXPECT_THROW(stream.can_read(data.size(), "Test message"), std::runtime_error);
}

TEST_F(onnx2_stream_2, NotEnd) {
  EXPECT_TRUE(stream.not_end());

  // Consommer tous les octets sauf un
  stream.read_bytes(data.size() - 1);
  EXPECT_TRUE(stream.not_end());

  // Consommer le dernier octet
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

// Tests pour StringWriteStream
TEST(onnx2_stream, StringWriteStream) {
  utils::StringWriteStream stream;

  // Tester l'écriture d'un variant uint64
  stream.write_variant_uint64(150);

  // Tester l'écriture d'un int64
  stream.write_int64(42);

  // Tester l'écriture d'un int32
  stream.write_int32(24);

  // Tester l'écriture d'un float
  stream.write_float(3.14f);

  // Tester l'écriture d'un double
  stream.write_double(2.71828);

  // Tester l'écriture d'un field header
  stream.write_field_header(10, 2);

  // Tester l'écriture d'une string
  stream.write_string("hello");

  // Vérifier la taille finale
  EXPECT_GT(stream.size(), 0);

  // Vérifier que les données ne sont pas nulles
  EXPECT_NE(stream.data(), nullptr);

  // Maintenant, lire les données écrites
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

// Tests pour les méthodes d'écriture de string
TEST(onnx2_stream, StringWriteStreamStrings) {
  utils::StringWriteStream stream;

  // Écrire une std::string
  std::string stdStr = "standard string";
  stream.write_string(stdStr);

  // Écrire une String
  utils::String str("custom string", 13);
  stream.write_string(str);

  // Écrire une RefString
  utils::RefString refStr("reference string", 16);
  stream.write_string(refStr);

  // Maintenant, lire les données écrites
  utils::StringStream readStream(stream.data(), stream.size());

  utils::RefString read1 = readStream.next_string();
  EXPECT_EQ(read1, "standard string");

  utils::RefString read2 = readStream.next_string();
  EXPECT_EQ(read2, "custom string");

  utils::RefString read3 = readStream.next_string();
  EXPECT_EQ(read3, "reference string");
}

// Tests pour BorrowedWriteStream
TEST(onnx2_stream, BorrowedWriteStream) {
  std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
  utils::BorrowedWriteStream stream(data.data(), data.size());

  // Vérifier que les données sont accessibles
  EXPECT_EQ(stream.size(), 5);
  EXPECT_EQ(stream.data(), data.data());

  // La méthode write_raw_bytes doit lever une exception
  EXPECT_THROW(stream.write_raw_bytes(nullptr, 0), std::runtime_error);
}

// Tests pour l'imbrication de StringWriteStream
TEST(onnx2_stream, NestedStringWriteStreams) {
  utils::StringWriteStream innerStream;

  // Écrire des données dans le flux intérieur
  innerStream.write_string("inner data");

  utils::StringWriteStream outerStream;

  // Écrire un en-tête de champ
  outerStream.write_field_header(15, 2);

  // Écrire le flux intérieur dans le flux extérieur
  outerStream.write_string_stream(innerStream);

  // Lire les données à partir du flux extérieur
  utils::StringStream readStream(outerStream.data(), outerStream.size());

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 15);
  EXPECT_EQ(field.wire_type, 2);

  // Lire le flux intérieur
  utils::StringStream innerReadStream;
  readStream.read_string_stream(innerReadStream);

  utils::RefString str = innerReadStream.next_string();
  EXPECT_EQ(str, "inner data");
}

// Tests pour les méthodes next_packed_element
TEST(onnx2_stream, NextPackedElement) {
  std::vector<uint8_t> data = {// un float: 3.14
                               0xC3, 0xF5, 0x48, 0x40,
                               // un int32: 42 (stocké en format binaire direct)
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
  // Tester next_uint64 avec des données insuffisantes
  std::vector<uint8_t> badData = {0x80, 0x80, 0x80}; // Varint invalide (continue indéfiniment)
  utils::StringStream badStream(badData.data(), badData.size());

  EXPECT_THROW(badStream.next_uint64(), std::runtime_error);

  // Tester can_read avec une taille trop grande
  std::vector<uint8_t> smallData = {0x01, 0x02};
  utils::StringStream smallStream(smallData.data(), smallData.size());

  EXPECT_THROW(smallStream.can_read(3, "Test message"), std::runtime_error);
}

TEST(onnx2_proto, StringStringEntryProto_Basic) {
  StringStringEntryProto entry;

  // Test des propriétés par défaut
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

  // Sérialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  // Vérification
  EXPECT_EQ(entry2.key(), "test_key");
  EXPECT_EQ(entry2.value(), "test_value");
}

// Tests pour IntIntListEntryProto
TEST(onnx2_proto, IntIntListEntryProto_Basic) {
  IntIntListEntryProto entry;

  // Test des propriétés par défaut
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

  // Sérialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  IntIntListEntryProto entry2;
  entry2.ParseFromString(serialized);

  // Vérification
  EXPECT_EQ(entry2.key(), 42);
  EXPECT_EQ(entry2.value().size(), 3);
  EXPECT_EQ(entry2.value()[0], 1);
  EXPECT_EQ(entry2.value()[1], 2);
  EXPECT_EQ(entry2.value()[2], 3);
}

// Tests pour TensorAnnotation
TEST(onnx2_proto, TensorAnnotation_Basic) {
  TensorAnnotation annotation;

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
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

  // Test des propriétés par défaut
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::UNDEFINED);
  EXPECT_EQ(tensor.dims().size(), 0);
  EXPECT_TRUE(tensor.name().empty());

  // Test des setters
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(3);
  tensor.set_name("my_tensor");

  // Ajout de données
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

  // Test des propriétés par défaut du segment
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

  // Test des propriétés par défaut
  EXPECT_EQ(tensor.raw_data().size(), 0);

  // Préparation de données
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Copie dans raw_data
  tensor.raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.raw_data().data(), data.data(), data.size() * sizeof(float));

  // Test de la taille
  EXPECT_EQ(tensor.raw_data().size(), data.size() * sizeof(float));

  // Vérification des données
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

  // Sérialisation
  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Vérification
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

  // Test des propriétés par défaut
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
  sparse.indices().dims().values.push_back(2); // Nombre d'éléments non nuls
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

  // Test des propriétés par défaut
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
  // Test de création d'un TensorProto
  TensorProto tensor;
  tensor.set_name("test_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(3);

  // Ajout de données
  for (int i = 0; i < 6; ++i) {
    tensor.float_data().values.push_back(static_cast<float>(i + 1));
  }

  // Vérification des propriétés
  EXPECT_EQ(tensor.name(), "test_tensor");
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 2);
  EXPECT_EQ(tensor.dims()[1], 3);
  EXPECT_EQ(tensor.float_data().size(), 6);
}

// Test pour vérifier la sérialisation/désérialisation d'un TensorProto
TEST(onnx2_proto, SerializeDeserializeTensorProto) {
  // Création d'un TensorProto
  TensorProto tensor1;
  tensor1.set_name("serialized_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.dims().values.push_back(2);
  tensor1.dims().values.push_back(2);
  tensor1.float_data().values.push_back(1.0f);
  tensor1.float_data().values.push_back(2.0f);
  tensor1.float_data().values.push_back(3.0f);
  tensor1.float_data().values.push_back(4.0f);

  // Sérialisation
  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Vérification que les deux tenseurs sont identiques
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

// Test pour vérifier la création et la manipulation d'un TypeProto
TEST(onnx2_proto, TypeProtoOperations) {
  // Création d'un TypeProto
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

  // Vérification
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), 1);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim().size(), 2);
  EXPECT_EQ(type.tensor_type().shape().dim()[0].dim_value(), 3);
  EXPECT_EQ(type.tensor_type().shape().dim()[1].dim_param(), "batch_size");
}

// Test pour vérifier la création et la sérialisation d'un StringStringEntryProto
TEST(onnx2_proto, StringStringEntryProtoOperations) {
  // Création d'un StringStringEntryProto
  StringStringEntryProto entry;
  entry.set_key("metadata_key");
  entry.set_value("metadata_value");

  // Vérification des propriétés
  EXPECT_EQ(entry.key(), "metadata_key");
  EXPECT_EQ(entry.value(), "metadata_value");

  // Sérialisation
  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  // Vérification
  EXPECT_EQ(entry2.key(), "metadata_key");
  EXPECT_EQ(entry2.value(), "metadata_value");
}

// Test pour vérifier la création et la manipulation d'un TensorProto avec raw_data
TEST(onnx2_proto, TensorProtoWithRawData) {
  // Création d'un TensorProto
  TensorProto tensor;
  tensor.set_name("raw_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.dims().values.push_back(2);
  tensor.dims().values.push_back(2);

  // Préparation des données
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Copie dans raw_data
  tensor.raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.raw_data().data(), data.data(), data.size() * sizeof(float));

  // Vérification
  EXPECT_EQ(tensor.name(), "raw_data_tensor");
  EXPECT_EQ(tensor.data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 2);
  EXPECT_EQ(tensor.dims()[1], 2);
  EXPECT_EQ(tensor.raw_data().size(), data.size() * sizeof(float));

  // Vérification du contenu de raw_data
  const float *raw_data_ptr = reinterpret_cast<const float *>(tensor.raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

// Test pour vérifier la création et manipulation d'un SparseTensorProto
TEST(onnx2_proto, SparseTensorProtoOperations) {
  // Création d'un SparseTensorProto
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
  sparse.indices().dims().values.push_back(2); // Nombre d'éléments non nuls
  sparse.indices().dims().values.push_back(2); // Nombre de dimensions
  sparse.indices().int64_data().values.push_back(0);
  sparse.indices().int64_data().values.push_back(2);
  sparse.indices().int64_data().values.push_back(1);
  sparse.indices().int64_data().values.push_back(3);

  // Vérification
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

  // Sérialisation
  std::string serialized;
  sparse.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  SparseTensorProto sparse2;
  sparse2.ParseFromString(serialized);

  // Vérification après désérialisation
  EXPECT_EQ(sparse2.dims().size(), 2);
  EXPECT_EQ(sparse2.values().float_data().size(), 2);
  EXPECT_EQ(sparse2.indices().int64_data().size(), 4);
}

// Test pour vérifier les manipulations d'un TensorShapeProto
TEST(onnx2_proto, TensorShapeProtoOperations) {
  // Création d'un TensorShapeProto
  TensorShapeProto shape;

  // Ajout de dimensions
  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  // Vérification
  EXPECT_EQ(shape.dim().size(), 2);
  EXPECT_TRUE(shape.dim()[0].has_dim_value());
  EXPECT_EQ(shape.dim()[0].dim_value(), 5);
  EXPECT_FALSE(shape.dim()[0].has_dim_param());

  EXPECT_FALSE(shape.dim()[1].has_dim_value());
  EXPECT_EQ(shape.dim()[1].dim_param(), "N");
  EXPECT_EQ(shape.dim()[1].denotation(), "batch");

  // Sérialisation
  std::string serialized;
  shape.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Désérialisation
  TensorShapeProto shape2;
  shape2.ParseFromString(serialized);

  // Vérification après désérialisation
  EXPECT_EQ(shape2.dim().size(), 2);
  EXPECT_EQ(shape2.dim()[0].dim_value(), 5);
  EXPECT_EQ(shape2.dim()[1].dim_param(), "N");
  EXPECT_EQ(shape2.dim()[1].denotation(), "batch");
}

// Test pour vérifier le comportement avec différents types de données dans TensorProto
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
  ASSERT_EQ(1, result.size());
  std::string serialized = result[0];
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
  ASSERT_EQ(1, result.size());
  std::string serialized = result[0];
  EXPECT_TRUE(serialized.find("my_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale_tensor") != std::string::npos);
}