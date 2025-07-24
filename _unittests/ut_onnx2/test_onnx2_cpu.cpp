#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"

using namespace onnx2;

void test_serialization_StringStringEntryProto() {
  StringStringEntryProto proto;
  proto.key() = "key__";
  proto.value() = "value__";
  ASSERT_EQUAL(proto.key(), "key__");
  ASSERT_EQUAL(proto.value(), "value__");
  std::string serie;
  proto.SerializeToString(serie);
  StringStringEntryProto proto2;
  proto2.ParseFromString(serie);
  ASSERT_EQUAL(proto.key(), proto2.key());
  ASSERT_EQUAL(proto.value(), proto2.value());
  std::string serie2;
  proto2.SerializeToString(serie2);
  ASSERT_EQUAL(serie, serie2);
}

int main(int, char**) {
  test_serialization_StringStringEntryProto();
}
