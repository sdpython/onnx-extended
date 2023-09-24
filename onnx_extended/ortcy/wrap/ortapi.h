#pragma once

#include "ortapi_version.h"
#include <stdexcept>
#include <string>
#include <vector>

#define OrtSessionType void

namespace ortapi {

inline std::size_t ElementSize(ONNXTensorElementDataType elem_type) {
  switch (elem_type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 2;
  default:
    throw std::runtime_error("One element type is not implemented in function "
                             "`ortapi::ElementSize()`.");
  }
}

inline std::size_t ElementSizeI(int elem_type) {
  return ElementSize((ONNXTensorElementDataType)elem_type);
}

class OrtShape {
private:
  int64_t size_;
  int64_t dims_[8];

public:
  inline OrtShape() { size_ = 0; }
  inline OrtShape(std::size_t ndim) { init(ndim); }
  inline void init(std::size_t ndim) {
    if (ndim > 8)
      throw std::runtime_error("shape cannot have more than 8 dimensions.");
    size_ = ndim;
  }
  inline int64_t ndim() const { return size_; }
  inline void set(std::size_t i, int64_t dim) { dims_[i] = dim; }
  inline const int64_t *dims() const { return dims_; }
};

class OrtCpuValue {
private:
  std::size_t size_;
  int elem_type_; // ONNXTensorElementDataType
  void *data_;
  void *ort_value_;

public:
  inline OrtCpuValue() {
    elem_type_ = -1;
    size_ = 0;
    ort_value_ = nullptr;
    data_ = nullptr;
  }
  inline void init(std::size_t size, int elem_type, void *data, void *ort_value) {
    size_ = size;
    elem_type_ = elem_type;
    data_ = data;
    ort_value_ = ort_value;
  }
  inline std::size_t size() { return size_; }
  inline int elem_type() { return elem_type_; }
  inline void *data() { return data_; }
  void free_ort_value();
};

// Simplified API for this project.
// see https://onnxruntime.ai/docs/api/c/

typedef void release(std::size_t output, int elem_type, std::size_t size, OrtShape *shape,
                     void *data, void *args);

std::vector<std::string> get_available_providers();

OrtSessionType *create_session();
void delete_session(OrtSessionType *);
void session_load_from_file(OrtSessionType *, const char *filename);
void session_load_from_bytes(OrtSessionType *, const void *buffer, std::size_t size);
void session_initialize(OrtSessionType *ptr, const char *optimized_file_path,
                        int graph_optimization_level = -1, int enable_cuda = 0,
                        int cuda_device_id = 0, int set_denormal_as_zero = 0,
                        int intra_op_num_threads = -1,
                        int inter_op_num_threads = -1,
                        char **custom_libs = nullptr);
size_t session_get_input_count(OrtSessionType *);
size_t session_get_output_count(OrtSessionType *);
size_t session_run(OrtSessionType *ptr, std::size_t n_inputs, OrtShape *shapes,
                   OrtCpuValue *values, std::size_t max_outputs,
                   OrtShape *out_shapes, OrtCpuValue *out_values);

} // namespace ortapi
