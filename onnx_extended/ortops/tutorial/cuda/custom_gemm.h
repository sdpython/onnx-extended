#pragma once

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

typedef enum _EpiloqueGemmKernel {
  Default = 0,
  Relu = 1,
  Gelu = 2
} EpiloqueGemmKernel;

struct CustomGemmKernel {
  CustomGemmKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  void set(const std::vector<int64_t> &shape_a,
           const std::vector<int64_t> &shape_b, int &M, int &N, int &K,
           int &lda, int &ldb, int &ldd, int row_major) const;

  void ComputeRowMajor(Ort::KernelContext &ctx, int n_inputs, bool has_bias,
                       bool has_scales, bool has_scales_Y,
                       Ort::ConstValue &input_A, Ort::ConstValue &input_B,
                       Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
                       Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y);
  void ComputeColMajor(Ort::KernelContext &ctx, int n_inputs, bool has_bias,
                       bool has_scales, bool has_scales_Y,
                       Ort::ConstValue &input_A, Ort::ConstValue &input_B,
                       Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
                       Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y);
  void ComputeGemm(
      Ort::KernelContext &ctx, int n_inputs, bool has_bias, bool has_scales,
      bool has_scales_Y, ONNXTensorElementDataType dtype_A,
      ONNXTensorElementDataType dtype_b, ONNXTensorElementDataType dtype_c,
      ONNXTensorElementDataType dtype_Y, const std::vector<int64_t> &shape_A,
      const std::vector<int64_t> &shape_B, const std::vector<int64_t> &shape_C,
      const std::vector<int64_t> &shape_Y, bool transa, bool transb,
      const void *p_input_a, const void *p_input_b, const void *p_input_c,
      const void *p_scale_a, const void *p_scale_b, const void *p_scale_y,
      void *p_output_y, int M, int N, int K, int lda, int ldb, int ldd);

  float alpha_;
  float beta_;
  // float beta_;
  bool transA_;
  bool transB_;
  bool fastAccumulationMode_;
  int64_t rowMajor_;
  int64_t smCount_;
  cublasComputeType_t computeType_;
  EpiloqueGemmKernel epilogue_;
};

struct CustomGemmOp : Ort::CustomOpBase<CustomGemmOp, CustomGemmKernel> {
  typedef Ort::CustomOpBase<CustomGemmOp, CustomGemmKernel> parent_type;
  CustomGemmOp(const char *op_name, ONNXTensorElementDataType ab_type,
               ONNXTensorElementDataType c_type,
               ONNXTensorElementDataType d_type)
      : parent_type() {
    op_name_ = op_name;
    ab_type_ = ab_type;
    c_type_ = c_type;
    d_type_ = d_type;
  }
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t index) const;

  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t index) const;

private:
  const char *op_name_;
  ONNXTensorElementDataType ab_type_;
  ONNXTensorElementDataType c_type_;
  ONNXTensorElementDataType d_type_;
};

} // namespace ortops
