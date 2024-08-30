// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library

#include <mutex>
#include <vector>

#include "ort_optim_cuda_lib.h"
#include "ortapi_version.h"

#include "add_or_mul_shared_input.h"
#include "addaddaddmulmulmul.h"
#include "addaddmulmul.h"
#include "addmul.h"
#include "mul_mul_sigmoid.h"
#include "mul_sigmoid.h"
#include "negxplus1.h"
#include "replace_zero.h"
#include "rotary.h"
#include "scatter_nd_of_shape.h"
#include "scatter_nd_of_shape_masked.h"
#include "submul.h"
#include "transpose_cast_2d.h"
#include "tri_matrix.h"

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
  static ortops::AddOrMulSharedInputOp<float, false> c_MulSharedOp32;
  static ortops::AddOrMulSharedInputOp<half, false> c_MulSharedOp16;
  static ortops::AddOrMulSharedInputOp<float, true> c_AddSharedOp32;
  static ortops::AddOrMulSharedInputOp<half, true> c_AddSharedOp16;

  static ortops::AddMulOp<float, false> c_MulAddOp32;
  static ortops::AddMulOp<half, false> c_MulAddOp16;
  static ortops::AddMulOp<float, true> c_AddMulOp32;
  static ortops::AddMulOp<half, true> c_AddMulOp16;

  static ortops::SubMulOp<float, false> c_MulSubOp32;
  static ortops::SubMulOp<half, false> c_MulSubOp16;
  static ortops::SubMulOp<float, true> c_SubMulOp32;
  static ortops::SubMulOp<half, true> c_SubMulOp16;

  static ortops::AddAddMulMulOp<float, false> c_MulMulOp32;
  static ortops::AddAddMulMulOp<half, false> c_MulMulOp16;
  static ortops::AddAddMulMulOp<float, true> c_AddAddOp32;
  static ortops::AddAddMulMulOp<half, true> c_AddAddOp16;

  static ortops::AddAddAddMulMulMulOp<float, false> c_MulMulMulOp32;
  static ortops::AddAddAddMulMulMulOp<half, false> c_MulMulMulOp16;
  static ortops::AddAddAddMulMulMulOp<float, true> c_AddAddAddOp32;
  static ortops::AddAddAddMulMulMulOp<half, true> c_AddAddAddOp16;

  static ortops::MulSigmoidOp<float> c_MulSigmoidOp32;
  static ortops::MulSigmoidOp<half> c_MulSigmoidOp16;

  static ortops::MulMulSigmoidOp<float> c_MulMulSigmoidOp32;
  static ortops::MulMulSigmoidOp<half> c_MulMulSigmoidOp16;

  static ortops::NegXplus1Op<float> c_NegXplus1Op32;
  static ortops::NegXplus1Op<half> c_NegXplus1Op16;
  static ortops::NegXplus1Op<int32_t> c_NegXplus1Opi32;

  static ortops::ReplaceZeroOp<float> c_ReplaceZeroOp32;
  static ortops::ReplaceZeroOp<half> c_ReplaceZeroOp16;

  static ortops::RotaryOp<float> c_RotaryOp32;
  static ortops::RotaryOp<half> c_RotaryOp16;

  static ortops::ScatterNDOfShapeOp<float> c_ScatterNDOfShapeOp32;
  static ortops::ScatterNDOfShapeOp<half> c_ScatterNDOfShapeOp16;

  static ortops::MaskedScatterNDOfShapeOp<float> c_MaskedScatterNDOfShapeOp32;
  static ortops::MaskedScatterNDOfShapeOp<half> c_MaskedScatterNDOfShapeOp16;

  static ortops::Transpose2DCastOp c_Transpose2DCast16(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  static ortops::Transpose2DCastOp c_Transpose2DCast32(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  static ortops::TriMatrixOp<float> c_TriMatrixOp32;
  static ortops::TriMatrixOp<half> c_TriMatrixOp16;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_AddSharedOp32);
    domain.Add(&c_AddSharedOp16);
    domain.Add(&c_MulSharedOp32);
    domain.Add(&c_MulSharedOp16);

    domain.Add(&c_AddMulOp32);
    domain.Add(&c_AddMulOp16);
    domain.Add(&c_MulAddOp32);
    domain.Add(&c_MulAddOp16);

    domain.Add(&c_AddAddOp32);
    domain.Add(&c_AddAddOp16);
    domain.Add(&c_MulMulOp32);
    domain.Add(&c_MulMulOp16);

    domain.Add(&c_SubMulOp32);
    domain.Add(&c_SubMulOp16);
    domain.Add(&c_MulSubOp32);
    domain.Add(&c_MulSubOp16);

    domain.Add(&c_AddAddAddOp32);
    domain.Add(&c_AddAddAddOp16);
    domain.Add(&c_MulMulMulOp32);
    domain.Add(&c_MulMulMulOp16);

    domain.Add(&c_MulSigmoidOp32);
    domain.Add(&c_MulSigmoidOp16);

    domain.Add(&c_MulMulSigmoidOp32);
    domain.Add(&c_MulMulSigmoidOp16);

    domain.Add(&c_NegXplus1Op32);
    domain.Add(&c_NegXplus1Op16);
    domain.Add(&c_NegXplus1Opi32);

    domain.Add(&c_ReplaceZeroOp32);
    domain.Add(&c_ReplaceZeroOp16);

    domain.Add(&c_RotaryOp32);
    domain.Add(&c_RotaryOp16);

    domain.Add(&c_ScatterNDOfShapeOp32);
    domain.Add(&c_ScatterNDOfShapeOp16);

    domain.Add(&c_MaskedScatterNDOfShapeOp32);
    domain.Add(&c_MaskedScatterNDOfShapeOp16);

    domain.Add(&c_Transpose2DCast16);
    domain.Add(&c_Transpose2DCast32);

    domain.Add(&c_TriMatrixOp32);
    domain.Add(&c_TriMatrixOp16);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
