#pragma once

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#define OrtAllocatorType void
#define OrtEnvType void
#define OrtMemoryInfoType void
#define OrtSessionInfoType void
#define OrtSessionOptionsType void
#define OrtSessionType void

namespace ortapi {

OrtEnvType* ort_create_env();
void ort_delete_env(OrtEnvType*);

OrtSessionOptionsType* ort_create_session_options();
void ort_delete_session_options(OrtSessionOptionsType*);

OrtSessionType* ort_create_session(const char* filename, OrtEnvType* env, OrtSessionOptionsType* sess_options);
void ort_delete_session(OrtSessionType* session);

OrtMemoryInfoType* ort_create_memory_info_cpu();
void ort_delete_memory_info(OrtMemoryInfoType*);

size_t ort_get_input_count(OrtSessionType*);
size_t ort_get_output_count(OrtSessionType*);

class OrtShape {
private:
  int64_t size_;
  int64_t dims_[8];

public:
  inline OrtShape(const std::vector<int> &shape) {
    if (shape.size() > 8)
      throw std::runtime_error("shape cannot have more than 8 dimensions.");
    size_ = static_cast<int64_t>(shape.size());
    memcpy(dims_, shape.data(), size_ * sizeof(int64_t));
  }
  inline int64_t ndim() const { return size_; }
  inline const int64_t *dims() const { return dims_; }
};

typedef enum {
    None=0,
    CPU=1,
    CUDA=2
} OrtProvider ;

class OrtValue {
    private:
        OrtProvider provider_;
        Ort::Value* ptr_ov;
    public:
        int element_type() const;
        inline OrtProvider provider() const { return provider_; }
        inline Ort::Value& value() const { return *ptr_ov; }

};

OrtSessionInfoType* ort_create_session_info(OrtSessionType*);
void ort_delete_session_info(OrtSessionType*);

} // namespace ortapi
