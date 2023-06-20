#include "common/common_kernels_cuda.h"
#include "custom_gemm.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

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

size_t CustomGemmOpFloat8E4M3FN::GetInputTypeCount() const { return 5; };

ONNXTensorElementDataType
CustomGemmOpFloat8E4M3FN::GetInputType(size_t index) const {
  switch (index) {
  case 0: // A
  case 1: // B
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
  case 2: // scale A
  case 3: // scale B
  case 4: // scale Y
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
};

size_t CustomGemmOpFloat8E4M3FN::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType
CustomGemmOpFloat8E4M3FN::GetOutputType(size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

#endif

///////////////////
// CustomGemmKernel
///////////////////

CustomGemmKernel::CustomGemmKernel(const OrtApi &api,
                                   const OrtKernelInfo *info) {
  row_major_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "rowMajor", true);
  transA_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transA", false);
  transB_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transB", false);
  fastAccumulationMode_ = KernelInfoGetOptionalAttributeInt64AsBool(
      api, info, "fastAccumulationMode", true);
  smCount_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "smCount", 0);
  alpha_ = KernelInfoGetOptionalAttribute<float>(api, info, "alpha", 1);

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

void CustomGemmKernel::set(const std::vector<int64_t> &a_shape,
                           const std::vector<int64_t> &b_shape, int &M, int &N,
                           int &K, int &lda, int &ldb, int &ldd) const {
  constexpr int ir = 0;
  constexpr int ic = 1 - ir;
  if (transA_ && !transB_) { // TN
    M = a_shape[ic];
    N = b_shape[ic];
    K = a_shape[ir];
    lda = a_shape[row_major_ ? ic : ir];
    ldb = b_shape[row_major_ ? ic : ir];
    ldd = b_shape[row_major_ ? ic : ir];
  } else if (!transA_ && !transB_) { // NN
    M = a_shape[ir];
    N = b_shape[ic];
    K = a_shape[ic];
    lda = a_shape[row_major_ ? ic : ir];
    ldb = b_shape[row_major_ ? ic : ir];
    ldd = b_shape[row_major_ ? ic : ir];
  } else if (!transA_ && transB_) { // NT
    M = a_shape[ir];
    N = b_shape[ir];
    K = a_shape[ic];
    lda = a_shape[row_major_ ? ic : ir];
    ldb = b_shape[row_major_ ? ic : ir];
    ldd = b_shape[row_major_ ? ir : ic];
  } else { // TT
    M = a_shape[ic];
    N = b_shape[ir];
    K = a_shape[ir];
    lda = a_shape[row_major_ ? ir : ic];
    ldb = b_shape[row_major_ ? ir : ic];
    ldd = b_shape[row_major_ ? ic : ir];
  }
}

void CustomGemmKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue scale_A, scale_B, scale_Y;

  auto memA = input_A.GetTensorMemoryInfo();
  EXT_ENFORCE(memA.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input A is not on CUDA");
  auto memB = input_B.GetTensorMemoryInfo();
  EXT_ENFORCE(memB.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input B is not on CUDA");

  int n_inputs = ctx.GetInputCount();
  if (n_inputs == 5) {
    scale_A = ctx.GetInput(2);
    scale_B = ctx.GetInput(3);
    scale_Y = ctx.GetInput(4);
    auto memsA = scale_A.GetTensorMemoryInfo();
    EXT_ENFORCE(memsA.GetDeviceType() ==
                    OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                "Scale A is not on CUDA");
    auto memsB = scale_B.GetTensorMemoryInfo();
    EXT_ENFORCE(memsB.GetDeviceType() ==
                    OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                "Scale B is not on CUDA");
    auto memsY = scale_Y.GetTensorMemoryInfo();
    EXT_ENFORCE(memsB.GetDeviceType() ==
                    OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                "Scale Y is not on CUDA");
  } else if (n_inputs != 2) {
    EXT_THROW("Number of inputs must be 2 or 5.");
  }

  std::vector<int64_t> a_shape = input_A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> b_shape = input_B.GetTensorTypeAndShapeInfo().GetShape();

  EXT_ENFORCE(a_shape.size() == 2);
  EXT_ENFORCE(b_shape.size() == 2);

  auto dtype_A =  input_A.GetTensorTypeAndShapeInfo().GetElementType();
auto dtype_B =  input_B.GetTensorTypeAndShapeInfo().GetElementType();

  int M, N, K, lda, ldb, ldd;
  set(a_shape, b_shape, M, N, K, lda, ldb, ldd);
  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  ONNXTensorElementDataType out_dtype =
      Y.GetTensorTypeAndShapeInfo().GetElementType();
  auto memY = Y.GetTensorMemoryInfo();
  EXT_ENFORCE(memY.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Output 1 is not on CUDA");

  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
  cublasLtHandle_t cublasLt;
  CUBLAS_THROW_IF_ERROR(cublasLtCreate(&cublasLt));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_cuda_type = ToCudaDataType(dtype_A);
  cudaDataType_t b_cuda_type = ToCudaDataType(dtype_B);
  cudaDataType_t d_cuda_type = ToCudaDataType(out_dtype);
  cudaDataType_t bias_cuda_type =
      ToCudaDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  cudaDataType_t scale_cuda_type = bias_cuda_type;

  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Adesc, a_cuda_type, transA_ ? K : M, transA_ ? M : K, lda));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(
      &Bdesc, b_cuda_type, transB_ ? N : K, transB_ ? K : N, ldb));
  CUBLAS_THROW_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Ddesc, d_cuda_type, M, N, ldd));

  if (row_major_) {

    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
  }

  CUBLAS_THROW_IF_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, computeType_, scale_cuda_type));
  cublasOperation_t transa = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  if (smCount_ != 0) {
    int math_sm_count = static_cast<int>(smCount_);
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count,
        sizeof(math_sm_count)));
  }

  const void *p_scale_a = nullptr;
  const void *p_scale_b = nullptr;
  const void *p_scale_y = nullptr;
  if (n_inputs == 5) {
    // gemm float 8
    const int8_t ifast_accumulation_mode = fastAccumulationMode_ ? 1 : 0;
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc,
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &ifast_accumulation_mode, sizeof(ifast_accumulation_mode)));
    p_scale_a = scale_A.GetTensorRawData();
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p_scale_a,
        sizeof(p_scale_a)));
    p_scale_b = scale_B.GetTensorRawData();
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p_scale_b,
        sizeof(p_scale_b)));
    p_scale_y = scale_Y.GetTensorRawData();
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &p_scale_y,
        sizeof(p_scale_b)));

    // float 8
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
    if (out_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
        out_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2) {
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      CUBLAS_THROW_IF_ERROR(
          cublasLtMatrixLayoutCreate(&Cdesc, bias_cuda_type, M, N, ldd));
      CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_cuda_type,
          sizeof(bias_cuda_type)));
    } else {
      CUBLAS_THROW_IF_ERROR(
          cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
    }
  } else {
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
  }
#else
    // An output is still needed but it is not initialized.
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
#endif

  if (row_major_) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &matrixOrder, sizeof(matrixOrder)));
  }

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

  // See
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true
  // with H100).
  size_t workspaceSize = std::max(
      (size_t)1 << 20,
      (std::min((size_t)(1 << 24), (size_t)std::max(K * M, K * N) * 4) +
       16)); // suggested fixed value 24Mb
  workspaceSize -= workspaceSize % 16;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference,
                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspaceSize, sizeof(workspaceSize));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(
      cublasLt, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults);
  EXT_ENFORCE(
      returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to find any suitable algorithm due to ",
      cublasGetErrorEnum(cuda_status), ", returnedResults=", returnedResults,
      ", alpha=", alpha_,
      // ", beta=", beta_,
      ", n_inputs=", n_inputs, ", A_type=", CudaDataTypeToString(a_cuda_type),
      ", B_type=", CudaDataTypeToString(b_cuda_type),
      ", result_type=", CudaDataTypeToString(d_cuda_type),
      ", bias_type=", CudaDataTypeToString(bias_cuda_type),
      ", scale_type=", CudaDataTypeToString(scale_cuda_type),
      ", computeType=", CublasComputeTypeToString(computeType_),
      ", epilogue=", epilogue, ", smCount=", smCount_, ", transA=", transA_,
      ", transB=", transB_,
      ", fastAccumulationMode=", (fastAccumulationMode_ ? 1 : 0),
      ", a_shape=", a_shape[0], "x", a_shape[1], ", b_shape=", b_shape[0], "x",
      b_shape[1], ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb,
      ", ldd=", ldd, ", workspaceSize=", workspaceSize,
      ", rowMajor=", (row_major_ ? 1 : 0),
      ". Check NVIDIA documentation to see what combination is valid: ",
      "https://docs.nvidia.com/cuda/cublas/"
      "index.html?highlight=cublasLtMatmulAlgoGetHeuristic#"
      "cublasltmatmulalgogetheuristic.");

  void *workspace = nullptr;
  if (workspaceSize > 0) {
    CUDA_THROW_IF_ERROR(cudaMalloc((void **)&workspace, workspaceSize));
  }
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  float beta = 0;
  void *C = Y.GetTensorMutableRawData();
  CUBLAS_THROW_IF_ERROR(cublasLtMatmul(
      cublasLt, operationDesc, static_cast<const void *>(&alpha_), /* alpha */
      input_A.GetTensorRawData(),                                  /* A */
      Adesc, input_B.GetTensorRawData(),                           /* B */
      Bdesc, static_cast<const void *>(&beta),                     /* beta */
      C,                                                           /* C */
      Cdesc, Y.GetTensorMutableRawData(),                          /* Y */
      Ddesc, &heuristicResult.algo,                                /* algo */
      workspace,               /* workspace */
      workspaceSize, stream)); /* stream */
  if (workspaceSize > 0) {
    cudaFree(workspace);
  }

  CUBLAS_THROW_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Ddesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  CUBLAS_THROW_IF_ERROR(cublasLtDestroy(cublasLt));
}

} // namespace ortops
