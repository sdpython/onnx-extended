#include "custom_gemm.h"

namespace ortops {

void *CustomGemmOp::CreateKernel(const OrtApi &api,
                                 const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
};

const char *CustomGemmOp::GetName() const { return "CustomGemm"; };

const char *CustomGemmOp::GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

size_t CustomGemmOp::GetInputTypeCount() const { return 5; };

ONNXTensorElementDataType CustomGemmOp::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

size_t CustomGemmOp::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType CustomGemmOp::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

CustomGemmKernel::CustomGemmKernel(const OrtApi &api,
                                   const OrtKernelInfo *info) {
  ThrowOnError(api, api.KernelInfoGetAttribute_float(info, "alpha", &alpha_));
  ThrowOnError(api, api.KernelInfoGetAttribute_float(info, "beta", &beta_));

  int64_t temp;
  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "transA", &temp));
  transA_ = temp == 1;
  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "transB", &temp));
  transB_ = temp == 1;
  ThrowOnError(api, api.KernelInfoGetAttribute_int64( info, "fastAccumulationMode", &temp));
  fastAccumulationMode_ = temp == 1;
  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "smCount", &smCount_));

  // A string attribute.
  std::string compute_type;
  size_t size;
  OrtStatus *status = api.KernelInfoGetAttribute_string(info, "computeType", nullptr, &size);
  if (status == nullptr) {
    compute_type = "CUBLAS_COMPUTE_32F";
  } else {
    api.ReleaseStatus(status);
    compute_type.resize(size + 1);
    status = api.KernelInfoGetAttribute_string(info, "computeType", (char*)compute_type.c_str(), &size);
    if (status == nullptr) {
      compute_type = "CUBLAS_COMPUTE_32F";
    } else {
      auto error_code = api.GetErrorCode(status);
      api.ReleaseStatus(status);
      if (error_code != ORT_OK) {
        compute_type = "CUBLAS_COMPUTE_32F";
      }
    }
  }

  if (compute_type == "CUBLAS_COMPUTE_16F") {
    computeType_ = CUBLAS_COMPUTE_16F;
    scaleType_ = CUDA_R_16F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F") {
    computeType_ = CUBLAS_COMPUTE_32F;
    scaleType_ = CUDA_R_32F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16F") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_16F;
    scaleType_ = CUDA_R_16F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16BF") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_16BF;
    scaleType_ = CUDA_R_16BF;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_TF32") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_TF32;
    scaleType_ = CUDA_R_32F;
  } else {
    EXT_THROW("Unexpected value for compute_type: ", compute_type, ".");
  }
}

void CustomGemmKernel::Compute(OrtKernelContext *context) {
  /*
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  Ort::ConstValue input_Y = ctx.GetInput(1);
  const double *X = input_X.GetTensorData<double>();
  const double *Y = input_Y.GetTensorData<double>();

  // Setup output, which is assumed to have the same dimensions as the inputs.
  std::vector<int64_t> dimensions =
  input_X.GetTensorTypeAndShapeInfo().GetShape();

  Ort::UnownedValue output = ctx.GetOutput(0, dimensions);
  double *out = output.GetTensorMutableData<double>();

  const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

  // Do computation
  double cst = att_tensor_double[0] + cst + static_cast<double>(att_float) +
  static_cast<double>(att_int64) + static_cast<double>(att_string[0]);

  for (size_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i] + cst;
  }
  */
}

} // namespace ortops
