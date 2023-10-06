// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "ort_optim_cpu_lib.h"
#include "ortapi_version.h"
#include "tree_ensemble.h"
#include "tree_ensemble.hpp"

static const char *c_OpDomain = "onnx_extented.ortops.optim.cpu";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION_SUPPORTED));
  Ort::UnownedSessionOptions session_options(options);

  // An instance remaining available until onnxruntime unload the library.
  static ortops::TreeEnsembleRegressor<float, float, float>
      c_TreeEnsembleRegressor;
  static ortops::TreeEnsembleClassifier<float, float, float>
      c_TreeEnsembleClassifier;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_TreeEnsembleRegressor);
    domain.Add(&c_TreeEnsembleClassifier);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
