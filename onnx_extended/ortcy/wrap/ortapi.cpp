#include "ortapi.h"

namespace ortapi {

OrtEnvType* ort_create_env() {
    return (OrtEnvType*) new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ortcy");
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

} // namespace ortapi
