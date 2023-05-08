#pragma once

#include "onnxruntime_c_api.h"
#include <string>
#include <vector>
#include <stdexcept>

#define OrtSessionType void

namespace ortapi {

class OrtShape {
private:
    int64_t size_;
    int64_t dims_[8];

public:
    inline OrtShape() { size_ = 0; }
    inline OrtShape(size_t ndim) { init(ndim); }
    inline void init(size_t ndim) {
        if (ndim > 8)
            throw std::runtime_error("shape cannot have more than 8 dimensions.");
        size_ = ndim;
    }
    inline int64_t ndim() const { return size_; }
    inline void set(size_t i, int64_t dim) { dims_[i] = dim; }
    inline const int64_t *dims() const { return dims_; }
};

class OrtCpuValue {
    private:
        size_t size_;
        int elem_type_;  // ONNXTensorElementDataType
        void* data_;
    public:
        inline OrtCpuValue() { elem_type_ = -1; }
        inline void init(size_t size, int elem_type, void* data) {
            size_ = size;
            elem_type_ = elem_type;
            data_ = data;
        }
        inline size_t size() { return size_; }
        inline int elem_type() { return elem_type_; }
        inline void* data() { return data_; }
};


// Simplified API for this project.
// see https://onnxruntime.ai/docs/api/c/

std::vector<std::string> get_available_providers();

OrtSessionType *create_session();
void delete_session(OrtSessionType *);
void session_load_from_file(OrtSessionType*, const char* filename);
void session_load_from_bytes(OrtSessionType*, const void* buffer, size_t size);
void session_initialize(OrtSessionType* ptr,
                        const char* optimized_file_path,
                        int graph_optimization_level = -1,
                        int enable_cuda = 0,
                        int cuda_device_id = 0,
                        int set_denormal_as_zero = 0,
                        int intra_op_num_threads = -1,
                        int inter_op_num_threads = -1);
size_t session_get_input_count(OrtSessionType *);
size_t session_get_output_count(OrtSessionType *);
size_t session_run(OrtSessionType* ptr,
                   size_t n_inputs,
                   OrtShape* shapes,
                   OrtCpuValue* values,
                   size_t max_outputs,
                   OrtCpuValue* out_ptr);

} // namespace ortapi
