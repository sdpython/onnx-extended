#include "ortapi.h"
#include "helpers.h"
#include <stdexcept>
#include <string>
#include <vector>

// https://onnxruntime.ai/docs/api/c/

namespace ortapi {

inline const OrtApi* GetOrtApi() {
    static const OrtApi* api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api_;
}

const char* ort_version() {
    return "GetOrtApi()->GetBuildInfoString();";
}

inline void ThrowOnError(OrtStatus* ort_status) {
    if (ort_status) {
        std::string message(GetOrtApi()->GetErrorMessage(ort_status));
        OrtErrorCode code = GetOrtApi()->GetErrorCode(ort_status);
        throw std::runtime_error(MakeString(
            "One call to onnxruntime failed due to (", code, ") ", message));
    }
}

inline std::vector<std::string> GetAvailableProviders() {
  int len;
  char** providers;
  ThrowOnError(GetOrtApi()->GetAvailableProviders(&providers, &len));
  std::vector<std::string> available_providers(providers, providers + len);
  ThrowOnError(GetOrtApi()->ReleaseAvailableProviders(providers, len));
  return available_providers;
}

class OrtInference {
public:

    OrtInference() {
        ThrowOnError(GetOrtApi()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ortcy", &env_));
        ThrowOnError(GetOrtApi()->CreateSessionOptions(&sess_options_));
        ThrowOnError(GetOrtApi()->CreateRunOptions(&run_options_));
        ThrowOnError(GetOrtApi()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info_));
        sess_ = nullptr;
        cpu_allocator_ = nullptr;
    }

    void LoadFromFile(const char* filepath) {
        EXT_ENFORCE(filepath != nullptr);
        EXT_ENFORCE(env_ != nullptr);
        EXT_ENFORCE(sess_options_ != nullptr);
        #ifdef _WIN32
        std::string name(filepath);
        std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
        std::wstring wname(cvt.from_bytes(name);
        ThrowOnError(GetOrtApi()->CreateSession(env_, wname.c_str(), sess_options_, &sess_));
        #else
        ThrowOnError(GetOrtApi()->CreateSession(env_, filepath, sess_options_, &sess_));
        #endif
        LoadFinalize();
    }

    void LoadFromBytes(const void* model_data, size_t model_data_length) {
        ThrowOnError(GetOrtApi()->CreateSessionFromArray(env_, model_data, model_data_length, sess_options_, &sess_));
        LoadFinalize();
    }

    ~OrtInference() {
        if (cpu_allocator_ != nullptr) GetOrtApi()->ReleaseAllocator(cpu_allocator_);
        if (sess_ != nullptr) GetOrtApi()->ReleaseSession(sess_);
        GetOrtApi()->ReleaseSessionOptions(sess_options_);
        GetOrtApi()->ReleaseRunOptions(run_options_);
        GetOrtApi()->ReleaseMemoryInfo(cpu_memory_info_);
        GetOrtApi()->ReleaseEnv(env_);
    }

    size_t GetInputCount() const { return n_inputs_; }
    size_t GetOutputCount() const { return n_outputs_; }

protected:
    void LoadFinalize() {
        EXT_ENFORCE(cpu_memory_info_ != nullptr);
        ThrowOnError(GetOrtApi()->CreateAllocator(sess_, cpu_memory_info_ , &cpu_allocator_));
        EXT_ENFORCE(cpu_allocator_ != nullptr);
        ThrowOnError(GetOrtApi()->SessionGetInputCount(sess_, &n_inputs_));
        ThrowOnError(GetOrtApi()->SessionGetOutputCount(sess_, &n_outputs_));
        input_names_.reserve(n_inputs_);
        output_names_.reserve(n_outputs_);
        
        char* name;
        for(size_t i = 0; i < n_inputs_; ++i) {
            ThrowOnError(GetOrtApi()->SessionGetInputName(sess_, i, cpu_allocator_, &name));
            input_names_.emplace_back(std::string(name));
            ThrowOnError(GetOrtApi()->AllocatorFree(cpu_allocator_, name));
        }
        for(size_t i = 0; i < n_outputs_; ++i) {
            ThrowOnError(GetOrtApi()->SessionGetOutputName(sess_, i, cpu_allocator_, &name));
            output_names_.emplace_back(std::string(name));
            ThrowOnError(GetOrtApi()->AllocatorFree(cpu_allocator_, name));
        }
    }

private:
    // before loading the model
    OrtEnv* env_;
    OrtSessionOptions* sess_options_;
    OrtRunOptions* run_options_;
    OrtMemoryInfo* cpu_memory_info_;

private:
    // after loading the model
    OrtSession* sess_;
    OrtAllocator* cpu_allocator_;
    size_t n_inputs_;
    size_t n_outputs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

//////// SIMPLE API //////

OrtSessionType* create_session() { return (OrtSessionType*)(new OrtInference()); }
void delete_session(OrtSessionType* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("Cannot delete a null pointer (delete_session).");
    delete (OrtInference*)ptr;
}
void session_load_from_file(OrtSessionType* ptr, const char* filename) { ((OrtInference*)ptr)->LoadFromFile(filename); }
size_t get_input_count(OrtSessionType* ptr) { return ((OrtInference*)ptr)->GetInputCount(); }
size_t get_output_count(OrtSessionType* ptr) { return ((OrtInference*)ptr)->GetOutputCount(); }




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
        //Ort::Value* ptr_ov;
    public:
        int element_type() const;
        inline OrtProvider provider() const { return provider_; }
        //inline Ort::Value& value() const { return *ptr_ov; }

};

} // namespace ortapi
