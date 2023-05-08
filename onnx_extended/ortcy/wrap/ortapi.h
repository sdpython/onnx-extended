#pragma once

#include "onnxruntime_c_api.h"
#include <string>
#include <vector>

#define OrtSessionType void

namespace ortapi {

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
size_t get_input_count(OrtSessionType *);
size_t get_output_count(OrtSessionType *);

// Simplified API for this project.
// see https://onnxruntime.ai/docs/api/c/

} // namespace ortapi
