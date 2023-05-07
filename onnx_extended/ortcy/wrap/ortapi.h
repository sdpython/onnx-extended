#pragma once

#include "onnxruntime_c_api.h"

#define OrtSessionType void

namespace ortapi {

OrtSessionType *create_session();
void delete_session(OrtSessionType *);
void session_load_from_file(OrtSessionType*, const char* filename);

size_t get_input_count(OrtSessionType *);
size_t get_output_count(OrtSessionType *);

// Simplified API for this project.
// see https://onnxruntime.ai/docs/api/c/

} // namespace ortapi
