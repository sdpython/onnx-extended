#include "common/common_kernels_cuda.h"
#include "custom_gemm.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// see
// https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublasLt.h

namespace ortops {

////////////////////
// CustomGemmOpFloat
////////////////////

void *CustomGemmOpFloat::CreateKernel(const OrtApi &api,
                                      const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
};

const char *CustomGemmOpFloat::GetName() const { return "CustomGemmFloat"; };

const char *CustomGemmOpFloat::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
};

size_t CustomGemmOpFloat::GetInputTypeCount() const { return 2; };

ONNXTensorElementDataType CustomGemmOpFloat::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomGemmOpFloat::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType CustomGemmOpFloat::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

///////////////////////////
// CustomGemmOpFloat8E4M3FN
///////////////////////////

#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080

void *CustomGemmOpFloat8E4M3FN::CreateKernel(const OrtApi &api,
                                             const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
};

const char *CustomGemmOpFloat8E4M3FN::GetName() const {
  return "CustomGemmFloat8E4M3FN";
};

const char *CustomGemmOpFloat8E4M3FN::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
};

size_t CustomGemmOpFloat8E4M3FN::GetInputTypeCount() const { return 4; };

ONNXTensorElementDataType
CustomGemmOpFloat8E4M3FN::GetInputType(size_t index) const {
  switch (index) {
  case 0: // A
  case 1: // B
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
  case 2: // scale A
  case 3: // scale B
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
};

size_t CustomGemmOpFloat8E4M3FN::GetOutputTypeCount() const { return 2; };

ONNXTensorElementDataType
CustomGemmOpFloat8E4M3FN::GetOutputType(size_t index) const {
  // D, scale D
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};

#endif

///////////////////
// CustomGemmKernel
///////////////////

CustomGemmKernel::CustomGemmKernel(const OrtApi &api,
                                   const OrtKernelInfo *info) {
  ThrowOnError(api, api.KernelInfoGetAttribute_float(info, "alpha", &alpha_));
  // ThrowOnError(api, api.KernelInfoGetAttribute_float(info, "beta", &beta_));
  transA_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transA", false);
  transB_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transB", false);
  fastAccumulationMode_ = KernelInfoGetOptionalAttributeInt64AsBool(
      api, info, "fastAccumulationMode", true);
  smCount_ = KernelInfoGetOptionalAttributeInt64(api, info, "smCount", 0);

  // A string attribute.
  std::string compute_type = KernelInfoGetOptionalAttributeString(
      api, info, "computeType", "CUBLAS_COMPUTE_32F");
  if (compute_type == "CUBLAS_COMPUTE_16F") {
    computeType_ = CUBLAS_COMPUTE_16F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F") {
    computeType_ = CUBLAS_COMPUTE_32F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16F") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_16F;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16BF") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_16BF;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_TF32") {
    computeType_ = CUBLAS_COMPUTE_32F_FAST_TF32;
  } else {
    EXT_THROW("Unexpected value for compute_type '", compute_type, "'.");
  }
}

void CustomGemmKernel::set(int M, int N, int K, int &lda, int &ldb,
                           int &ldd) const {
  if (transA_ && !transB_) { // TN
    lda = K;
    ldb = K;
    ldd = M;
  } else if (!transA_ && !transB_) { // NN
    lda = M;
    ldb = K;
    ldd = M;
  } else if (!transA_ && transB_) { // NT
    lda = M;
    ldb = N;
    ldd = M;
  } else { // TT
    EXT_THROW("transA_ == true && transB_ == true not allowed.");
  }
}

void CustomGemmKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue scale_A, scale_B;
  int n_inputs = ctx.GetInputCount();
  if (n_inputs == 4) {
    scale_A = ctx.GetInput(2);
    scale_B = ctx.GetInput(3);
  } else if (n_inputs != 2) {
    EXT_THROW("Number of inputs must be 2 or 4.");
  }

  std::vector<int64_t> a_shape = input_A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> b_shape = input_B.GetTensorTypeAndShapeInfo().GetShape();

  EXT_ENFORCE(a_shape.size() == 2);
  EXT_ENFORCE(b_shape.size() == 2);

  ONNXTensorElementDataType dtypes[4] = {
      input_A.GetTensorTypeAndShapeInfo().GetElementType(),
      input_B.GetTensorTypeAndShapeInfo().GetElementType(),
      n_inputs == 4 ? scale_A.GetTensorTypeAndShapeInfo().GetElementType()
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      n_inputs == 4 ? scale_B.GetTensorTypeAndShapeInfo().GetElementType()
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
  };

  int M, N, K;
  if (transA_) {
    M = a_shape[1];
    K = a_shape[0];
  } else {
    M = a_shape[0];
    K = a_shape[1];
  }

  N = transB_ ? b_shape[0] : b_shape[1];
  EXT_ENFORCE(M >= 0 && K > 0 && N >= 0);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  ONNXTensorElementDataType out_dtype =
      Y.GetTensorTypeAndShapeInfo().GetElementType();

  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
  cublasLtHandle_t cublasLt;
  CUBLAS_THROW_IF_ERROR(cublasLtCreate(&cublasLt));

  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  int lda, ldb, ldd;
  set(M, N, K, lda, ldb, ldd);

  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(B) x op(A) +
  // beta * C
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_type = ToCudaDataType(dtypes[0]);
  cudaDataType_t b_type = ToCudaDataType(dtypes[1]);
  cudaDataType_t d_type = ToCudaDataType(out_dtype);
  cudaDataType_t bias_type =
      ToCudaDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  cudaDataType_t scale_dtype = bias_type;

  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Adesc, a_type, transA_ ? M : K, transA_ ? K : M, lda));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Bdesc, b_type, transB_ ? K : N, transB_ ? N : K, ldb));
  CUBLAS_THROW_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Cdesc, bias_type, M, N, ldd));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, d_type, M, N, ldd));

  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(
      Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));

  if (n_inputs == 4) {
    // gemm float 8

    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        scale_A.GetTensorRawData(), sizeof(float)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        scale_B.GetTensorRawData(), sizeof(float)));
  }

  cublasLtMatmulDescCreate(&operationDesc, computeType_, scale_dtype);
  cublasOperation_t transa = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb, sizeof(transb));
  const int8_t ifast_accumulation_mode = fastAccumulationMode_ ? 0 : 1;
  cublasLtMatmulDescSetAttribute(
      operationDesc,
      cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
      &ifast_accumulation_mode, sizeof(ifast_accumulation_mode));
  /*
  if (has_C) {
    cublasLtMatmulDescSetAttribute(operationDesc,
                                   cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                   &bias_type,
                                   sizeof(bias_type));
  */

  if (n_inputs == 4) {
    // float 8
    std::vector<int64_t> scale_dimensions{1};
    Ort::UnownedValue scale_Y = ctx.GetOutput(1, scale_dimensions);
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        scale_Y.GetTensorRawData(), sizeof(float)));
  }

  if (smCount_ != 0) {
    int math_sm_count = static_cast<int>(smCount_);
    cublasLtMatmulDescSetAttribute(operationDesc,
                                   CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                   &math_sm_count, sizeof(math_sm_count));
  }

  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

  cublasLtMatmulPreferenceCreate(&preference);

  // See
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true
  // with H100). size_t type_size = std::max(std::max(TypeSize(dtypes[0]),
  // TypeSize(dtypes[1])), std::max(std::max(TypeSize(dtypes[2]),
  // TypeSize(dtypes[3])), TypeSize(dtypes[4])));
  size_t workspaceSize = std::max(
      (size_t)1 << 20,
      (std::min((size_t)(1 << 24), (size_t)std::max(K * M, K * N) * 4) +
       16)); // suggested fixed value 24Mb
  workspaceSize -= workspaceSize % 16;
  cublasLtMatmulPreferenceSetAttribute(preference,
                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspaceSize, sizeof(workspaceSize));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(
      cublasLt, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults);
  EXT_ENFORCE(returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
              " Unable to find any suitable algorithm due to ",
              cublasGetErrorEnum(cuda_status), ", preference=", preference,
              ", returnedResults=", returnedResults, ", alpha=", alpha_,
              // ", beta=", beta_,
              ", A_type=", CudaDataTypeToString(ToCudaDataType(dtypes[0])),
              ", B_type=", CudaDataTypeToString(ToCudaDataType(dtypes[1])),
              ", C_type=", CudaDataTypeToString(ToCudaDataType(dtypes[2])),
              ", result_type=", CudaDataTypeToString(ToCudaDataType(dtypes[4])),
              ", bias_type=", CudaDataTypeToString(bias_type),
              ", scale_type=", CudaDataTypeToString(scale_dtype),
              ", computeType=", CublasComputeTypeToString(computeType_),
              ", epilogue=", epilogue, ", smCount=", smCount_,
              ", transA=", transA_, ", transB=", transB_,
              ", fastAccumulationMode=", (fastAccumulationMode_ ? 1 : 0),
              ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb,
              ", ldd=", ldd, ", workspaceSize=", workspaceSize,
              ". Check NVIDIA documentation to see what combination is valid: ",
              "https://docs.nvidia.com/cuda/cublas/"
              "index.html?highlight=cublasLtMatmulAlgoGetHeuristic#"
              "cublasltmatmulalgogetheuristic.");
  void *workspace = nullptr;
  if (workspaceSize > 0) {
    cudaMalloc((void **)&workspace, workspaceSize);
  }
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  float beta = 0;
  CUBLAS_THROW_IF_ERROR(cublasLtMatmul(
      cublasLt, operationDesc, static_cast<const void *>(&alpha_),  /* alpha */
      input_A.GetTensorRawData(),                                   /* A */
      Adesc, input_B.GetTensorRawData(),                            /* B */
      Bdesc, static_cast<const void *>(&beta),                      /* beta */
      nullptr,                                                      /* C */
      Cdesc, Y.GetTensorMutableRawData(),                           /* Y */
      Ddesc, &heuristicResult.algo,                                 /* algo */
      workspace,                                                    /* workspace */
      workspaceSize, stream));                                      /* stream */
  if (workspaceSize > 0) {
    cudaFree(workspace);
  }

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatmulDescDestroy(operationDesc);
  CUBLAS_THROW_IF_ERROR(cublasLtDestroy(cublasLt));
}

} // namespace ortops
