// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "ort_optim_cpu_lib.h"
#include "ort_sparse.hpp"
#include "ort_svm.hpp"
#include "ort_tfidf_vectorizer.hpp"
#include "ort_tree_ensemble.hpp"
#include "ortapi_version.h"

static const char *c_OpDomain = "onnx_extended.ortops.optim.cpu";

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
  static ortops::DenseToSparse<float> c_DenseToSparse;
  static ortops::SparseToDense<float> c_SparseToDense;
  static ortops::SVMClassifier<float> c_SVMClassifier;
  static ortops::SVMRegressor<float> c_SVMRegressor;
  static ortops::TreeEnsembleRegressor<onnx_c_ops::DenseFeatureAccessor<float>, float, float>
      c_TreeEnsembleRegressor;
  static ortops::TreeEnsembleClassifier<onnx_c_ops::DenseFeatureAccessor<float>, float, float>
      c_TreeEnsembleClassifier;
  static ortops::TreeEnsembleRegressor<onnx_c_ops::SparseFeatureAccessor<float>, float, float>
      c_TreeEnsembleRegressorSparse;
  static ortops::TreeEnsembleClassifier<onnx_c_ops::SparseFeatureAccessor<float>, float, float>
      c_TreeEnsembleClassifierSparse;
  static ortops::TfIdfVectorizer<int64_t, float> c_TfIdfVectorizer;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_DenseToSparse);
    domain.Add(&c_SparseToDense);
    domain.Add(&c_SVMClassifier);
    domain.Add(&c_SVMRegressor);
    domain.Add(&c_TreeEnsembleClassifier);
    domain.Add(&c_TreeEnsembleClassifierSparse);
    domain.Add(&c_TreeEnsembleRegressor);
    domain.Add(&c_TreeEnsembleRegressorSparse);
    domain.Add(&c_TfIdfVectorizer);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
