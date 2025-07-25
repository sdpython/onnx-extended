#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
#include <gtest/gtest.h>

using namespace onnx2;

TEST(onnx2, RefString) {
  utils::RefString a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
}

TEST(onnx2, String) {
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

TEST(onnx2, TensorProtoName1) {
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

TEST(onnx2, TensorProtoName2) {
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
  std::string check2 = tp.name().as_string();
  EXPECT_EQ(name, check2);
}

TEST(onnx2, TensorProtoNameStringToString1) {
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

TEST(onnx2, TensorProtoNameStringToString2) {
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

TEST(onnx2, TensorProtoName00) { TensorProto tp; }
TEST(onnx2, TensorProtoName01) {
  TensorProto tp;
  tp.set_name("rt");
}

TEST(onnx2, serialization_StringStringEntryProto) {
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

TEST(onnx2, TensorShapeProto1) {
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
