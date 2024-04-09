// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "addaddmulmul.h"
#include "ort_optim_cuda_lib.h"
#include "ortapi_version.h"
#include "scatter_nd_of_shape.h"

static const char *c_OpDomain = "onnx_extended.ortops.optim.cuda";

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

  // Instances remaining available until onnxruntime unload the library.
  static ortops::AddAddMulMulOp<float, false> c_MulMulOp32;
  static ortops::AddAddMulMulOp<half, false> c_MulMulOp16;
  static ortops::AddAddMulMulOp<float, true> c_AddAddOp32;
  static ortops::AddAddMulMulOp<half, true> c_AddAddOp16;
  static ortops::ScatterNDOfShapeOp<float> c_ScatterNDOfShapeOp32;
  static ortops::ScatterNDOfShapeOp<half> c_ScatterNDOfShapeOp16;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_AddAddOp32);
    domain.Add(&c_AddAddOp16);
    domain.Add(&c_MulMulOp32);
    domain.Add(&c_MulMulOp16);
    domain.Add(&c_ScatterNDOfShapeOp32);
    domain.Add(&c_ScatterNDOfShapeOp16);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
