// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "../../ortops_version.h"
#include "custom_gemm.h"
#include "ort_tutorial_cuda_lib.h"

static const char *c_OpDomain = "onnx_extented.ortops.tutorial.cuda";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION_ALLOWED));
  Ort::UnownedSessionOptions session_options(options);

  // An instance remaining available until onnxruntime unload the library.
  static ortops::CustomGemmOp c_CustomGemmFloat(
      "CustomGemmFloat", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      false);
  static ortops::CustomGemmOp c_CustomGemmFloat16(
      "CustomGemmFloat16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, false);
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
  static ortops::CustomGemmOp c_CustomGemmFloat8E4M3FN(
      "CustomGemmFloat8E4M3FN", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      false);
  static ortops::CustomGemmOp c_CustomGemmFloat8E4M3FNTime(
      "CustomGemmFloat8E4M3FNTime", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      false);
#endif

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_CustomGemmFloat);
    domain.Add(&c_CustomGemmFloat16);
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
    domain.Add(&c_CustomGemmFloat8E4M3FN);
    domain.Add(&c_CustomGemmFloat8E4M3FNTime);
#endif

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
