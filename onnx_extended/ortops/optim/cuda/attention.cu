#include "attention.h"
#include "common/c_op_helpers.h"
#include "common/common_kernels.h"
#include "cuda/common_kernels_cuda.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ortops {

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

//////////////////
// AttentionOp...
// https://github.com/microsoft/onnxruntime/pull/25684
//////////////////

template <typename T1, typename T2, typename U>
void *AttentionOp<T1, T2, U>::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<AttentionKernel<T1, T2, U>>(api, info).release();
}

template <typename T1, typename T2, typename U> const char *AttentionOp<T1, T2, U>::GetName() const {
  return "Attention";
}

template <typename T1, typename T2, typename U>
const char *AttentionOp<T1, T2, U>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T1, typename T2, typename U> size_t AttentionOp<T1, T2, U>::GetInputTypeCount() const {
  return 6;
};

template <typename T1, typename T2, typename U>
ONNXTensorElementDataType AttentionOp<T1, T2, U>::GetInputType(std::size_t index) const {
  switch(index) {
    case 0:
    case 1:
    case 4:
      return CTypeToOnnxType<T1>().onnx_type();
    case 2:
    case 5: 
      return CTypeToOnnxType<T2>().onnx_type();
    case 3:
      return CTypeToOnnxType<U>().onnx_type();
  }
}

template <typename T1, typename T2, typename U>
OrtCustomOpInputOutputCharacteristic
AttentionOp<T1, T2, U>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 0:
  case 1:
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

template <typename T1, typename T2, typename U> size_t AttentionOp<T1, T2, U>::GetOutputTypeCount() const {
  return 4;
}

template <typename T1, typename T2, typename U>
ONNXTensorElementDataType AttentionOp<T1, T2, U>::GetOutputType(std::size_t index) const {
  switch(index) {
    case 0:
    case 1:
    case 3:
      return CTypeToOnnxType<T1>().onnx_type();
    case 2:
      return CTypeToOnnxType<T2>().onnx_type();
  }
}

template <typename T1, typename T2, typename U>
OrtCustomOpInputOutputCharacteristic
AttentionOp<T1, T2, U>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 1:
  case 2:
  case 3:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// AttentionKernel
///////////////////

template <typename T1, typename T2, typename U>
AttentionKernel<T1, T2, U>::AttentionKernel(const OrtApi &api, const OrtKernelInfo *info) {
  is_causal_ = KernelInfoGetOptionalAttributeInt64AsBool(api, info, "is_causal", false);
  kv_num_heads_ = KernelInfoGetOptionalAttributeInt64(api, info, "kv_num_heads", 0);
  q_num_heads_ = KernelInfoGetOptionalAttributeInt64(api, info, "q_num_heads", 0);
  qk_matmul_output_mode_ = KernelInfoGetOptionalAttributeInt64AsBool(api, info, "qk_matmul_output_mode", false);
  scale_ = KernelInfoGetOptionalAttributeFloat(api, info, "scale", -1.0);
  softcap_ = KernelInfoGetOptionalAttributeInt64(api, info, "softcap", 0);
  softmax_precision_ = KernelInfoGetOptionalAttributeInt64(api, info, "softmax_precision", -1);
}

template <typename T1, typename T2, typename U>
void AttentionKernel<T1, T2, U>::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  EXT_ENFORCE(n_inputs == 3 || n_inputs == 4 || n_inputs == 6, "Expected 3, 4, 6 inputs not ", n_inputs, ".");
  Ort::ConstValue A = ctx.GetInput(0);
  Ort::ConstValue B = ctx.GetInput(1);
  Ort::ConstValue C = ctx.GetInput(2);
  Ort::UnownedValue output;
}

static AttentionOp<float, float, float> _attention32;
static AttentionOp<float, float, bool> _attention32_boolmask;

} // namespace ortops
