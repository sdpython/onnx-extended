#include "custom_gemm.h"
#include "common/common_kernels_cuda.h"
#include <cublasLt.h>
#include <cublas_v2.h>

// see https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublasLt.h

namespace ortops {

////////////////////
// CustomGemmOpFloat
////////////////////

void *CustomGemmOpFloat::CreateKernel(const OrtApi &api,
                                      const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
};

const char *CustomGemmOpFloat::GetName() const { return "CustomGemmFloat"; };

const char *CustomGemmOpFloat::GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

size_t CustomGemmOpFloat::GetInputTypeCount() const { return 5; };

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

#if ORT_VERSION >= 1160

void *CustomGemmOpFloat8E4M3FN::CreateKernel(const OrtApi &api,
                                      const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
};

const char *CustomGemmOpFloat8E4M3FN::GetName() const { return "CustomGemmFloat8E4M3FN"; };

const char *CustomGemmOpFloat8E4M3FN::GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

size_t CustomGemmOpFloat8E4M3FN::GetInputTypeCount() const { return 5; };

ONNXTensorElementDataType CustomGemmOpFloat8E4M3FN::GetInputType(size_t index) const {
  switch (index) {
    case 0:  // A
    case 1:  // B
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
    case 2:  // C
    case 3:  // bias
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case 4:  // result
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32;
    default:
      EXT_THROW("index=", index, " is out of boundary.");
  }
};

size_t CustomGemmOpFloat8E4M3FN::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType CustomGemmOpFloat8E4M3FN::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
};

#endif

///////////////////
// CustomGemmKernel
///////////////////

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
  OrtStatus *status = api.KernelInfoGetAttribute_int64(info, "smCount", &smCount_);
  if (status == nullptr) {
    smCount_ = 0;
  } else {
    auto error_code = api.GetErrorCode(status);
    api.ReleaseStatus(status);
    if (error_code != ORT_OK) {
      smCount_ = 0;
    }
  }

  // A string attribute.
  std::string compute_type;
  size_t size;
  status = api.KernelInfoGetAttribute_string(info, "computeType", nullptr, &size);
  if (status == nullptr) {
    compute_type = "CUBLAS_COMPUTE_32F";
  } else {
    auto error_code = api.GetErrorCode(status);
    api.ReleaseStatus(status);
    if (error_code == ORT_OK) {
      compute_type.resize(size + 1);
      status = api.KernelInfoGetAttribute_string(info, "computeType", (char*)compute_type.c_str(), &size);
      if (status == nullptr) {
        compute_type = "CUBLAS_COMPUTE_32F";
      } else {
        error_code = api.GetErrorCode(status);
        api.ReleaseStatus(status);
        if (error_code != ORT_OK) {
          compute_type = "CUBLAS_COMPUTE_32F";
        }
      }
    } else {
      compute_type = "CUBLAS_COMPUTE_32F";
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
    EXT_THROW("Unexpected value for compute_type '", compute_type, "'.");
  }
}

void CustomGemmKernel::set(int M, int N, int K, int& lda, int& ldb, int& ldd) const {
  if (transA_ && !transB_) {  // TN
    lda = K;
    ldb = K;
    ldd = M;
  }
  else if (!transA_ && !transB_) {  // NN
    lda = M;
    ldb = K;
    ldd = M;
  }
  else if (!transA_ && transB_) {  // NT
    lda = M;
    ldb = N;
    ldd = M;
  }
  else {  // TT
    EXT_THROW("transA_ == true && transB_ == true not allowed.");
  }
}

void CustomGemmKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue input_C = ctx.GetInput(2);
  Ort::ConstValue input_D = ctx.GetInput(3);
  Ort::ConstValue input_E = ctx.GetInput(4);

  std::vector<int64_t> a_shape = input_A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> b_shape = input_B.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> c_shape = input_C.GetTensorTypeAndShapeInfo().GetShape();

  EXT_ENFORCE(a_shape.size() == 2);
  EXT_ENFORCE(b_shape.size() == 2);
  EXT_ENFORCE(c_shape.size() == 0 || c_shape.size() == 2);

  ONNXTensorElementDataType dtypes[5] = {
    input_A.GetTensorTypeAndShapeInfo().GetElementType(),
    input_B.GetTensorTypeAndShapeInfo().GetElementType(),
    input_C.GetTensorTypeAndShapeInfo().GetElementType(),
    input_D.GetTensorTypeAndShapeInfo().GetElementType(),
    input_E.GetTensorTypeAndShapeInfo().GetElementType()
  };

  int M, N, K;
  if (transA_) {
    M = a_shape[1];
    K = a_shape[0];
  }
  else {
    M = a_shape[0];
    K = a_shape[1];
  }

  N = transB_ ? b_shape[0] : b_shape[1];
  EXT_ENFORCE(M >= 0 && K > 0 && N >= 0);

  std::vector<int64_t> dimensions {M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);

  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
  cublasLtHandle_t cublasLt;
  CUBLAS_THROW_IF_ERROR(cublasLtCreate(&cublasLt));

  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  int lda, ldb, ldd;
  set(M, N, K, lda, ldb, ldd);

  bool has_C = beta_ != 0;

  // broadcast bias if needed and is present
  if (has_C) {
    if (c_shape.size() == 1) {
      // if C is (), (1,) or (1, 1), broadcast the scalar
      EXT_THROW("Broadcasting is not implemented in GemmFloat8.");
    }
    else if (c_shape.size() == 1 || c_shape[0] == 1) {
      // C is (N,) or (1, N), broadcast using Y(N,M) = 1 * C(N,1) x ones(1,M) + 0 * C
      EXT_THROW("Broadcasting is not implemented in GemmFloat8.");
    }
    else if (b_shape.size() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * C
      EXT_THROW("Broadcasting is not implemented in GemmFloat8.");
    }
    else {
      // C is (M, N), no broadcast needed.
    }
  }

  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(B) x op(A) + beta * C
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_type = ToCudaDataType(dtypes[0]);
  cudaDataType_t b_type = ToCudaDataType(dtypes[1]);
  cudaDataType_t c_type = ToCudaDataType(dtypes[2]);
  cudaDataType_t d_type = ToCudaDataType(dtypes[4]);
  cudaDataType_t bias_type;
  // bool gemm_float8;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  if (a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3 || 
      a_type == CUDA_R_8F_E5M2 || b_type == CUDA_R_8F_E5M2) {
    bias_type = c_type == CUDA_R_16F ? CUDA_R_16F : CUDA_R_16BF;
    // gemm_float8 = true;
  }
  else {
#endif
    bias_type = c_type;
    // gemm_float8 = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  }
#endif
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, a_type, transA_ ? M : K, transA_ ? K : M, lda));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, b_type, transB_ ? K : N, transB_ ? N : K, ldb));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, c_type, M, N, ldd));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, d_type, M, N, ldd));

  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matrixOrder, sizeof(matrixOrder)));

  /*
  float scale_value = 1;
  if (gemm_float8) {
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                          &scale_value,
                                                          sizeof(scale_value)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                          &scale_value,
                                                          sizeof(scale_value)));    
  }
  */

  // CUDA_R_32F is the scale type for the time being since it is not used.
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescCreate#cublasltmatmuldesccreate
  cublasLtMatmulDescCreate(&operationDesc, computeType_, scaleType_);
  cublasOperation_t transa = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  const int8_t ifast_accumulation_mode = fastAccumulationMode_ ? 0 : 1;
  cublasLtMatmulDescSetAttribute(operationDesc,
                                 cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                 &ifast_accumulation_mode,
                                 sizeof(ifast_accumulation_mode));
  if (has_C) {
    cublasLtMatmulDescSetAttribute(operationDesc,
                                   cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                   &bias_type,
                                   sizeof(bias_type));
  }

  /*
  if (d_type == CUDA_R_8F_E4M3 || d_type == CUDA_R_8F_E5M2) {
      CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                            CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                            &scale_value,
                                                            sizeof(scale_value)));
      CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                            CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                            &scale_value,
                                                            sizeof(scale_value)));
  }
  */

  /*
  // TODO add inputs for the scales.
  // No scale for the time being so no need to set.
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER
  */

  if (smCount_ != 0) {
    int math_sm_count = static_cast<int>(smCount_);
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        &math_sm_count, sizeof(math_sm_count));
  }

  /*
  // No bias for the time being.
  if (relu_bias) {
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          relu_bias, sizeof(*relu_bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }
  */

  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  cublasLtMatmulPreferenceCreate(&preference);

  // See https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true with H100).
  // size_t type_size = std::max(std::max(TypeSize(dtypes[0]), TypeSize(dtypes[1])), std::max(std::max(TypeSize(dtypes[2]), TypeSize(dtypes[3])), TypeSize(dtypes[4])));
  size_t workspaceSize = std::max((size_t)1 << 20, (std::min((size_t)(1 << 24), (size_t)std::max(K * M, K * N) * 4) + 16));  // suggested fixed value 24Mb
  workspaceSize -= workspaceSize % 16;
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(cublasLt, operationDesc,
                                                              Adesc, Bdesc, Cdesc, Ddesc,
                                                              preference, 1, &heuristicResult, &returnedResults);
  EXT_ENFORCE(returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
              " Unable to find any suitable algorithm due to ", cublasGetErrorEnum(cuda_status),
              ", preference=", preference, ", returnedResults=", returnedResults,
              ", alpha=", alpha_, ", beta=", beta_,
              ", A_type=", CudaDataTypeToString(ToCudaDataType(dtypes[0])),
              ", B_type=", CudaDataTypeToString(ToCudaDataType(dtypes[1])),
              ", C_type=", CudaDataTypeToString(ToCudaDataType(dtypes[2])),
              ", result_type=", CudaDataTypeToString(ToCudaDataType(dtypes[4])),
              ", bias_type=", CudaDataTypeToString(bias_type),
              ", scale_type=", CudaDataTypeToString(scaleType_),
              ", computeType=", CublasComputeTypeToString(computeType_),
              ", transA=", transA_, ", transB=", transB_,
              ", fastAccumulationMode=", (fastAccumulationMode_ ? 1 : 0),
              ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb, ", ldd=", ldd,
              ", workspaceSize=", workspaceSize, ". Check NVDIDIA documentation to see what combination is valid: ",
              "https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic.");
  void* workspace = nullptr;
  if (workspaceSize > 0) {
    cudaMalloc((void**)&workspace, workspaceSize);
  }
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  CUBLAS_THROW_IF_ERROR(cublasLtMatmul(
                 cublasLt,
                 operationDesc,
                 static_cast<const void*>(&alpha_),             /* alpha */
                 input_A.GetTensorRawData(),                    /* A */
                 Adesc,
                 input_B.GetTensorRawData(),                    /* B */
                 Bdesc,
                 static_cast<const void*>(&beta_),              /* beta */
                 has_C ? input_C.GetTensorRawData() : nullptr,  /* C */
                 Cdesc,
                 Y.GetTensorMutableRawData(),                   /* Y */
                 Ddesc,
                 &heuristicResult.algo,                         /* algo */
                 workspace,                                     /* workspace */
                 workspaceSize,
                 stream));                                      /* stream */
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

  // #else
  // ORT_ENFORCE(false, "Compiling with CUDA_VERSION >= 11.8 is needed!");
  // #endif

}

} // namespace ortops
