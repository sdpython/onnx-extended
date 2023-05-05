#pragma once

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

namespace ortapi {

void *ort_create_env();
void ort_delete_env(void*);

void* ort_create_session_options();
void ort_delete_session_options(void*);

void* ort_create_session(const char* filename, void* env, void* sess_options);
void ort_delete_session(void *session);

class OrtShape {
private:
  int64_t size_;
  int64_t dims_[8];

public:
  inline OrtShape(const std::vector<int> &shape) {
    if (shape.size() > 8)
      throw std::runtime_error("shape cannot have move than 8 dimensions.");
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

} // namespace ortapi
