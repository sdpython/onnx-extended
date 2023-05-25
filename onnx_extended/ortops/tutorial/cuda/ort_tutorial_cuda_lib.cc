// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "ort_tutorial_cuda_lib.h"
#include "custom_gemm.h"

static const char* c_OpDomain = "onnx_extented.ortops.tutorial.cuda";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  Ort::UnownedSessionOptions session_options(options);

  // An instance remaining available until onnxruntime unload the library.
  static ortops::CustomGemm c_CustomGemm;

  OrtStatus* result = nullptr;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_CustomGemm);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  catch (const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }

  return result;
}
