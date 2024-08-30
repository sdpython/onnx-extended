// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "custom_gemm.h"
#include "custom_tree_assembly.h"
#include "dynamic_quantize_linear.h"
#include "my_kernel.h"
#include "my_kernel_attr.h"
#include "ort_tutorial_cpu_lib.h"
#include "ortapi_version.h"

static const char *c_OpDomain = "onnx_extended.ortops.tutorial.cpu";

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
  static ortops::MyCustomOp c_CustomOp;
  static ortops::MyCustomOpWithAttributes c_CustomOpAttr;
  static ortops::CustomGemmOp c_CustomGemmFloat(
      "CustomGemmFloat", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false);
  static ortops::CustomGemmOp c_CustomGemmFloat16(
      "CustomGemmFloat16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, false);
  static ortops::CustomTreeAssemblyOp c_CustomTreeAssembly(false);

#if ORT_API_VERSION_SUPPORTED >= 16
  static ortops::DynamicQuantizeLinearOp c_dql(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN);

  static ortops::CustomGemmOp c_CustomGemmFloat8E4M3FN(
      "CustomGemmFloat8E4M3FN", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false);
#endif

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_CustomOp);
    domain.Add(&c_CustomOpAttr);
    domain.Add(&c_CustomGemmFloat);
    domain.Add(&c_CustomGemmFloat16);
    domain.Add(&c_CustomTreeAssembly);
#if ORT_API_VERSION_SUPPORTED >= 16
    domain.Add(&c_dql);
    domain.Add(&c_CustomGemmFloat8E4M3FN);
#endif

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
