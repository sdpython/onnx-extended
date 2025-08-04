#include "onnx_helper.h"

namespace onnx2 {
bool IteratorTensorProto::next() {
  while (!positions_.empty()) {
    Position &pos = positions_.back();
    // loop over initializers
    if (pos.graph->ref_initializer().size() > 0) {
      if (pos.node_initializer_index < pos.graph->ref_initializer().size()) {
        tp_ = &(pos.graph->ref_initializer()[pos.node_initializer_index]);
        ++pos.node_initializer_index;
        return true;
      }
    }
    if (pos.graph->ref_node().size() > 0) {
      bool break_look = false;
      while (pos.node_index < pos.graph->ref_node().size()) {
        NodeProto *node = &(pos.graph->ref_node()[pos.node_index]);
        while (pos.attr_index < node->ref_attribute().size()) {
          AttributeProto &att = node->ref_attribute()[pos.attr_index];
          if (att.has_t()) {
            tp_ = &(att.ref_t());
            ++pos.attr_index;
            return true;
          } else if (att.has_g()) {
            GraphProto *subgraph = &(att.ref_g());
            positions_.emplace_back(Position{subgraph});
            break_look = true;
            ++pos.attr_index;
            break;
          }
          EXT_ENFORCE(!att.has_tensors(), "not implemented yet for attribute with tensors");
          EXT_ENFORCE(!att.has_graphs(), "not implemented yet for attribute with graphs");
          ++pos.attr_index;
        }
        if (break_look)
          break;
        ++pos.node_index;
        pos.attr_index = 0;
      }
    }
    positions_.pop_back();
  }
  return false;
}

void SaveOnnxModel(ModelProto &model, utils::BinaryWriteStream &stream, SerializeOptions &options) {
  if (stream.ExternalWeights()) {
    // Let's fill external data.
  }
  model.SerializeToStream(stream, options);
}

} // namespace onnx2
