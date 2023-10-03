#include "ortapi.h"
#include "onnx_extended_helpers.h"
#include "ortapi_inline.h"
#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

// https://onnxruntime.ai/docs/api/c/

namespace ortapi {

std::vector<std::string> get_available_providers() {
  int len;
  char **providers;
  ThrowOnError(GetOrtApi()->GetAvailableProviders(&providers, &len));
  std::vector<std::string> available_providers(providers, providers + len);
  ThrowOnError(GetOrtApi()->ReleaseAvailableProviders(providers, len));
  return available_providers;
}

void OrtCpuValue::free_ort_value() {
  if (ort_value_ != nullptr) {
    GetOrtApi()->ReleaseValue((OrtValue *)ort_value_);
    ort_value_ = nullptr;
  }
}

class OrtInference {
public:
  OrtInference() {
    ThrowOnError(
        GetOrtApi()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ortcy", &env_));
    ThrowOnError(GetOrtApi()->CreateSessionOptions(&sess_options_));
    ThrowOnError(GetOrtApi()->CreateRunOptions(&run_options_));
    ThrowOnError(GetOrtApi()->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info_));
    sess_ = nullptr;
    cpu_allocator_ = nullptr;
    n_inputs_ = 0;
    n_outputs_ = 0;
  }

  void LoadFromFile(const char *filepath) {
    EXT_ENFORCE(filepath != nullptr);
    EXT_ENFORCE(env_ != nullptr);
    EXT_ENFORCE(sess_options_ != nullptr);
#ifdef _WIN32
    std::string name(filepath);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
    std::wstring wname(cvt.from_bytes(name));
    ThrowOnError(
        GetOrtApi()->CreateSession(env_, wname.c_str(), sess_options_, &sess_));
#else
    ThrowOnError(
        GetOrtApi()->CreateSession(env_, filepath, sess_options_, &sess_));
#endif
    LoadFinalize();
  }

  void LoadFromBytes(const void *model_data, std::size_t model_data_length) {
    ThrowOnError(GetOrtApi()->CreateSessionFromArray(
        env_, model_data, model_data_length, sess_options_, &sess_));
    LoadFinalize();
  }

  ~OrtInference() {
    if (cpu_allocator_ != nullptr)
      GetOrtApi()->ReleaseAllocator(cpu_allocator_);
    if (sess_ != nullptr)
      GetOrtApi()->ReleaseSession(sess_);
    GetOrtApi()->ReleaseSessionOptions(sess_options_);
    GetOrtApi()->ReleaseRunOptions(run_options_);
    GetOrtApi()->ReleaseMemoryInfo(cpu_memory_info_);
    GetOrtApi()->ReleaseEnv(env_);
  }

  std::size_t GetInputCount() const { return n_inputs_; }
  std::size_t GetOutputCount() const { return n_outputs_; }

  void Initialize(const char *optimized_file_path = nullptr,
                  int graph_optimization_level = -1, int enable_cuda = 0,
                  int cuda_device_id = 0, int set_denormal_as_zero = 0,
                  int intra_op_num_threads = -1, int inter_op_num_threads = -1,
                  const char **custom_libs = nullptr) {
    if (graph_optimization_level != -1) {
      ThrowOnError(GetOrtApi()->SetSessionGraphOptimizationLevel(
          sess_options_, (GraphOptimizationLevel)graph_optimization_level));
    }
    if (optimized_file_path != nullptr) {
      std::string path(optimized_file_path);
      if (!path.empty()) {
#ifdef _WIN32
        std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
        std::wstring wpath(cvt.from_bytes(path));
        ThrowOnError(GetOrtApi()->SetOptimizedModelFilePath(sess_options_,
                                                            wpath.c_str()));
#else
        ThrowOnError(GetOrtApi()->SetOptimizedModelFilePath(sess_options_,
                                                            path.c_str()));
#endif
      }
    }
    if (enable_cuda) {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = cuda_device_id;
      cuda_options.do_copy_in_default_stream = true;
      // TODO: Support arena configuration for users of test runner
      ThrowOnError(GetOrtApi()->SessionOptionsAppendExecutionProvider_CUDA(
          sess_options_, &cuda_options));
    }
    // see https://github.com/microsoft/onnxruntime/blob/main/
    // include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    if (set_denormal_as_zero) {
      ThrowOnError(GetOrtApi()->AddSessionConfigEntry(
          sess_options_, "session.set_denormal_as_zero", "1"));
    }
    if (intra_op_num_threads != -1) {
      ThrowOnError(GetOrtApi()->SetIntraOpNumThreads(sess_options_,
                                                     intra_op_num_threads));
    }
    if (inter_op_num_threads != -1) {
      ThrowOnError(GetOrtApi()->SetInterOpNumThreads(sess_options_,
                                                     inter_op_num_threads));
    }
    if (custom_libs != nullptr) {
#ifdef _WIN32
      std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
#endif
      while (*custom_libs != nullptr) {
#ifdef _WIN32
        std::wstring wpath(cvt.from_bytes(*custom_libs));
        ThrowOnError(GetOrtApi()->RegisterCustomOpsLibrary_V2(sess_options_,
                                                              wpath.c_str()));
#else
        ThrowOnError(GetOrtApi()->RegisterCustomOpsLibrary_V2(sess_options_,
                                                              *custom_libs));
#endif
        ++custom_libs;
      }
    }
  }

  std::size_t Run(std::size_t n_inputs, OrtShape *shapes, OrtCpuValue *values,
             std::size_t max_outputs, OrtShape *out_shapes,
             OrtCpuValue *out_values) {
    if (max_outputs < n_outputs_)
      EXT_THROW("Not enough expected outputs, max_outputs=", (uint64_t)max_outputs, " > ",
                (uint64_t)n_outputs_, ".");
    if (n_inputs > n_inputs_)
      EXT_THROW("Too many inputs, n_inputs=", (uint64_t)n_inputs, " > ", (uint64_t)n_inputs, ".");
    std::vector<OrtValue *> ort_values(n_inputs);

    for (std::size_t i = 0; i < n_inputs; ++i) {
      ONNXTensorElementDataType elem_type =
          (ONNXTensorElementDataType)values[i].elem_type();
      ThrowOnError(GetOrtApi()->CreateTensorWithDataAsOrtValue(
          cpu_memory_info_, values[i].data(),
          values[i].size() * ElementSize(elem_type), shapes[i].dims(),
          shapes[i].ndim(), elem_type, &ort_values[i]));
    }

    std::vector<OrtValue *> ort_values_out(n_outputs_);
    ThrowOnError(GetOrtApi()->Run(sess_, run_options_, input_names_call_.data(),
                                  ort_values.data(), n_inputs,
                                  output_names_call_.data(), n_outputs_,
                                  ort_values_out.data()));

    for (std::size_t i = 0; i < n_inputs; ++i) {
      GetOrtApi()->ReleaseValue(ort_values[i]);
    }
    OrtTensorTypeAndShapeInfo *info;
    ONNXTensorElementDataType elem_type;
    std::size_t size, n_dims;
    void *data;
    for (std::size_t i = 0; i < n_outputs_; ++i) {
      ThrowOnError(
          GetOrtApi()->GetTensorTypeAndShape(ort_values_out[i], &info));
      ThrowOnError(GetOrtApi()->GetTensorElementType(info, &elem_type));
      if (elem_type ==
          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        GetOrtApi()->ReleaseTensorTypeAndShapeInfo(info);
        for (; i < n_outputs_; ++i) {
          GetOrtApi()->ReleaseValue(ort_values_out[i]);
        }
        throw std::runtime_error("tensor(string) is not supported as outputs.");
      }
      ThrowOnError(GetOrtApi()->GetTensorShapeElementCount(info, &size));
      ThrowOnError(GetOrtApi()->GetTensorMutableData(ort_values_out[i], &data));
      ThrowOnError(GetOrtApi()->GetDimensionsCount(info, &n_dims));
      out_shapes[i].init(n_dims);
      ThrowOnError(GetOrtApi()->GetDimensions(
          info, (int64_t *)out_shapes[i].dims(), n_dims));
      /* typedef void copy_allocate(std::size_t output, int elem_type, std::size_t size,
                                    OrtShape shape, void* data, void* args); */
      GetOrtApi()->ReleaseTensorTypeAndShapeInfo(info);
      out_values[i].init(size, elem_type, data, ort_values_out[i]);
      // GetOrtApi()->ReleaseValue(ort_values_out[i]);
    }
    return n_outputs_;
  }

protected:
  void LoadFinalize() {
    EXT_ENFORCE(cpu_memory_info_ != nullptr);
    ThrowOnError(
        GetOrtApi()->CreateAllocator(sess_, cpu_memory_info_, &cpu_allocator_));
    EXT_ENFORCE(cpu_allocator_ != nullptr);
    ThrowOnError(GetOrtApi()->SessionGetInputCount(sess_, &n_inputs_));
    ThrowOnError(GetOrtApi()->SessionGetOutputCount(sess_, &n_outputs_));
    input_names_.reserve(n_inputs_);
    output_names_.reserve(n_outputs_);

    char *name;
    for (std::size_t i = 0; i < n_inputs_; ++i) {
      ThrowOnError(
          GetOrtApi()->SessionGetInputName(sess_, i, cpu_allocator_, &name));
      input_names_.emplace_back(std::string(name));
      ThrowOnError(GetOrtApi()->AllocatorFree(cpu_allocator_, name));
    }
    for (std::size_t i = 0; i < n_outputs_; ++i) {
      ThrowOnError(
          GetOrtApi()->SessionGetOutputName(sess_, i, cpu_allocator_, &name));
      output_names_.emplace_back(std::string(name));
      ThrowOnError(GetOrtApi()->AllocatorFree(cpu_allocator_, name));
    }
    input_names_call_.resize(n_inputs_);
    for (std::size_t i = 0; i < n_inputs_; ++i) {
      input_names_call_[i] = input_names_[i].c_str();
    }
    output_names_call_.resize(n_inputs_);
    for (std::size_t i = 0; i < n_inputs_; ++i) {
      output_names_call_[i] = output_names_[i].c_str();
    }
  }

private:
  // before loading the model
  OrtEnv *env_;
  OrtSessionOptions *sess_options_;
  OrtRunOptions *run_options_;
  OrtMemoryInfo *cpu_memory_info_;

private:
  // after loading the model
  OrtSession *sess_;
  OrtAllocator *cpu_allocator_;
  std::size_t n_inputs_;
  std::size_t n_outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_names_call_;
  std::vector<const char *> output_names_call_;
};

/*
typedef enum {
    None=0,
    CPU=1,
    CUDA=2
} OrtProvider ;

*/

//////// SIMPLE API //////

OrtSessionType *create_session() {
  return (OrtSessionType *)(new OrtInference());
}
void delete_session(OrtSessionType *ptr) {
  if (ptr == nullptr)
    throw std::runtime_error("Cannot delete a null pointer (delete_session).");
  delete (OrtInference *)ptr;
}
void session_load_from_file(OrtSessionType *ptr, const char *filename) {
  ((OrtInference *)ptr)->LoadFromFile(filename);
}
void session_load_from_bytes(OrtSessionType *ptr, const void *buffer,
                             std::size_t size) {
  ((OrtInference *)ptr)->LoadFromBytes(buffer, size);
}
size_t session_get_input_count(OrtSessionType *ptr) {
  return ((OrtInference *)ptr)->GetInputCount();
}
size_t session_get_output_count(OrtSessionType *ptr) {
  return ((OrtInference *)ptr)->GetOutputCount();
}

void session_initialize(OrtSessionType *ptr, const char *optimized_file_path,
                        int graph_optimization_level, int enable_cuda,
                        int cuda_device_id, int set_denormal_as_zero,
                        int intra_op_num_threads, int inter_op_num_threads,
                        char **custom_libs) {
  ((OrtInference *)ptr)
      ->Initialize(optimized_file_path, graph_optimization_level, enable_cuda,
                   cuda_device_id, set_denormal_as_zero, intra_op_num_threads,
                   inter_op_num_threads, (const char **)custom_libs);
}

size_t session_run(OrtSessionType *ptr, std::size_t n_inputs, OrtShape *shapes,
                   OrtCpuValue *values, std::size_t max_outputs,
                   OrtShape *out_shapes, OrtCpuValue *out_values) {
  return ((OrtInference *)ptr)
      ->Run(n_inputs, shapes, values, max_outputs, out_shapes, out_values);
}

} // namespace ortapi
