#include "onnx_extended/onnx2/cpu/onnx2_helper.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace onnx2;
using namespace onnx2::utils;

TEST(onnx2_helper, IteratorTensorProto) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  IteratorTensorProto itp(&model.ref_graph());
  std::vector<uint8_t> dt;
  while (itp.next()) {
    dt.push_back(itp->ref_raw_data()[0]);
  }
  EXPECT_EQ(dt.size(), 2);
  EXPECT_EQ(dt[0], 2);
  EXPECT_EQ(dt[1], 1);
}

TEST(onnx2_helper, IteratorTensorProto_NestedGraph) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  IteratorTensorProto itp(&model.ref_graph());
  std::vector<uint8_t> dt;
  while (itp.next()) {
    dt.push_back(itp->ref_raw_data()[0]);
  }
  EXPECT_EQ(dt.size(), 4);
  EXPECT_EQ(dt[0], 2);
  EXPECT_EQ(dt[1], 4);
  EXPECT_EQ(dt[2], 3);
  EXPECT_EQ(dt[3], 1);
}

TEST(onnx2_helper, IteratorTensorProto_ExternalData) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  PopulateExternalData(model, 2, "external_data.bin");

  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    EXPECT_TRUE(it->has_external_data());
    EXPECT_EQ(it->ref_external_data().size(), 3);
  }

  ClearExternalData(model);
  while (it.next()) {
    EXPECT_FALSE(it->has_external_data());
  }
}

TEST(onnx2_helper, SerializeModelProtoToStream) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  SerializeOptions options;
  options.raw_data_threshold = 2;
  utils::TwoFilesWriteStream stream("SerializeModelProtoToStream.onnx",
                                    "SerializeModelProtoToStream.data");

  SerializeModelProtoToStream(model, stream, options);
}
