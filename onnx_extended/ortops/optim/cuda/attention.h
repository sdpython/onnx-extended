#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

template <typename T1, typename T2, typename U> struct AttentionKernel {
  AttentionKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
  private:
    bool is_causal_;
    int kv_num_heads_;
    int q_num_heads_;
    float scale_;
    float softcap_;
    int softmax_precision_;
};

template <typename T1, typename T2, typename U>
struct AttentionOp : Ort::CustomOpBase<AttentionOp<T1, T2, U>, AttentionKernel<T1, T2, U>> {
  typedef Ort::CustomOpBase<AttentionOp<T1, T2, U>, AttentionKernel<T1, T2, U>> parent_type;
  AttentionOp() : parent_type() {}
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

} // namespace ortops
