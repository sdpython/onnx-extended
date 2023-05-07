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

size_t get_input_count(OrtSessionType *);
size_t get_output_count(OrtSessionType *);

// Simplified API for this project.
// see https://onnxruntime.ai/docs/api/c/

} // namespace ortapi
