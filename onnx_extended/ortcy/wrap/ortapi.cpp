#include "ortapi.h"
// #include "onnxruntime_cxx_inline.h"

namespace ortapi {

OrtEnvType *ort_create_env() {
  return (OrtEnvType *)new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ortcy");
}

void ort_delete_env(OrtEnvType* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_env).");
    delete (Ort::Env*)ptr;
}

OrtSessionOptionsType* ort_create_session_options() {
    return (OrtSessionOptionsType*) new Ort::SessionOptions();
}

void ort_delete_session_options(OrtSessionOptionsType* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_session_options).");
    delete (Ort::SessionOptions*)ptr;
}

OrtSessionType* ort_create_session(const char* filename, OrtEnvType* env, OrtSessionOptionsType* sess_options) {
    if (sess_options == nullptr)
        throw std::runtime_error("sess_options cannot be null.");
    const auto& api = Ort::GetApi();
    #if _WIN32
    std::string name(filename);
    std::wstring(name.begin(), name.end());
    Ort::Session* session = new Ort::Session(*(Ort::Env*)env,
                                             (ORTCHAR_T*)name.c_str(),
                                             *(Ort::SessionOptions*)sess_options);
    #else
    Ort::Session* session = new Ort::Session(*(Ort::Env*)env,
                                             filename,
                                             *(Ort::SessionOptions*)sess_options);
    #endif
    return (OrtSessionType*) session;
}

void ort_delete_session(OrtSessionType *ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_session).");
    delete (Ort::Session*)ptr;
}

size_t ort_get_input_count(OrtSessionType* ptr) {
    return ((Ort::Session*)ptr)->GetInputCount();
}

size_t ort_get_output_count(OrtSessionType* ptr) {
    return ((Ort::Session*)ptr)->GetOutputCount();
}

OrtMemoryInfoType* ort_create_memory_info_cpu() {
    return (OrtMemoryInfoType*) new Ort::MemoryInfo("CPU", OrtArenaAllocator, 0, OrtMemTypeDefault);
}

void ort_delete_memory_info(OrtMemoryInfoType* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_memory_info).");
    delete (Ort::MemoryInfo*)ptr;
}


class OrtSessionInfo {
    public:
        OrtSessionInfo(OrtSessionType* ptr) {
            session_ = ptr;
            n_inputs_ = ((Ort::Session*)ptr)->GetInputCount();
            n_outputs_ = ((Ort::Session*)ptr)->GetOutputCount();
            input_names_.reserve(n_inputs_);
            output_names_.reserve(n_outputs_);
            Ort::AllocatorWithDefaultOptions ort_alloc;
            {
                for(size_t i = 0; i < n_inputs_; ++i) {
                    Ort::AllocatedStringPtr s = ((Ort::Session*)ptr)->GetInputNameAllocated(i, ort_alloc);
                    input_names_.emplace_back(s.get());
                }
                for(size_t i = 0; i < n_outputs_; ++i) {
                    Ort::AllocatedStringPtr s = ((Ort::Session*)ptr)->GetOutputNameAllocated(i, ort_alloc);
                    output_names_.emplace_back(s.get());
                }
            }
        }
    private:
        OrtSessionType* session_;
        size_t n_inputs_;
        size_t n_outputs_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
};

OrtSessionInfoType* ort_create_session_info(OrtSessionType* ptr) {
    return (OrtSessionInfoType*)(new OrtSessionInfo(ptr));
}

void ort_delete_session_info(OrtSessionType* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_session_info).");
    delete (OrtSessionInfo*)ptr;
}

} // namespace ortapi
