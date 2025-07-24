#include <gtest/gtest.h>
#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"

using namespace onnx2;

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
