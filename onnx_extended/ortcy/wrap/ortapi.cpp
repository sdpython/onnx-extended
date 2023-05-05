#include "ortapi.h"

namespace ortapi {

void *ort_create_env() {
    return (void*) new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ortcy");
}

void ort_delete_env(void* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_env).");
    delete (Ort::Env*)ptr;
}

void* ort_create_session_options() {
    return (void*) new Ort::SessionOptions();
}

void ort_delete_session_options(void* ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_session_options).");
    delete (Ort::SessionOptions*)ptr;
}

void* ort_create_session(const char* filename, void* env, void* sess_options) {
    if (sess_options == nullptr)
        throw std::runtime_error("sess_options cannot be null.");
    const auto& api = Ort::GetApi();
    Ort::Session* session = new Ort::Session(*(Ort::Env*)env, filename, *(Ort::SessionOptions*)sess_options);
    return (void*) session;
}

void ort_delete_session(void *ptr) {
    if (ptr == nullptr)
        throw std::runtime_error("ptr is null and cannot be deleted (ort_delete_session).");
    delete (Ort::Session*)ptr;
}

} // namespace ortapi
