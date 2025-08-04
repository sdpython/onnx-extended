#pragma once

#include "onnx2.h"

namespace onnx2 {

class IteratorTensorProto {
protected:
  struct Position {
    GraphProto *graph;
    int node_index = 0;
    int attr_index = 0;
    int node_initializer_index = 0;
  };

public:
  explicit inline IteratorTensorProto(GraphProto *graph) : tp_(nullptr), positions_() {
    positions_.emplace_back(Position{graph});
  }
  inline TensorProto &operator*() { return *tp_; }
  inline TensorProto *operator->() { return tp_; }
  bool next();

private:
  TensorProto *tp_;
  std::vector<Position> positions_;
};

void SaveOnnxModel(ModelProto &model, utils::BinaryWriteStream &stream, SerializeOptions &options);

} // namespace onnx2
