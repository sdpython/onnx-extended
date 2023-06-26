#include "common/common_kernels_cuda.h"
#include "custom_gemm.h"
#include <chrono>
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
}

const char *CustomGemmOpFloat::GetName() const { return "CustomGemmFloat"; }

const char *CustomGemmOpFloat::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

size_t CustomGemmOpFloat::GetInputTypeCount() const { return 3; }

OrtCustomOpInputOutputCharacteristic
CustomGemmOpFloat::GetInputCharacteristic(size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

ONNXTensorElementDataType CustomGemmOpFloat::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t CustomGemmOpFloat::GetOutputTypeCount() const { return 2; }

ONNXTensorElementDataType CustomGemmOpFloat::GetOutputType(size_t index) const {
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

//////////////////////
// CustomGemmOpFloat16
//////////////////////

void *CustomGemmOpFloat16::CreateKernel(const OrtApi &api,
                                        const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char *CustomGemmOpFloat16::GetName() const { return "CustomGemmFloat16"; }

const char *CustomGemmOpFloat16::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

size_t CustomGemmOpFloat16::GetInputTypeCount() const { return 2; }

ONNXTensorElementDataType
CustomGemmOpFloat16::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOpFloat16::GetInputCharacteristic(size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

size_t CustomGemmOpFloat16::GetOutputTypeCount() const { return 2; }

ONNXTensorElementDataType
CustomGemmOpFloat16::GetOutputType(size_t index) const {
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

///////////////////////////
// CustomGemmOpFloat8E4M3FN
///////////////////////////

#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080

void *CustomGemmOpFloat8E4M3FN::CreateKernel(const OrtApi &api,
                                             const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char *CustomGemmOpFloat8E4M3FN::GetName() const {
  return "CustomGemmFloat8E4M3FN";
}

const char *CustomGemmOpFloat8E4M3FN::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

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
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOpFloat8E4M3FN::GetInputCharacteristic(size_t index) const {
  switch (index) {
  case 0:
  case 1:
  case 3:
  case 4:
  case 5:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("index=", index, " is out of boundary.");
  }
}

size_t CustomGemmOpFloat8E4M3FN::GetOutputTypeCount() const { return 2; }

ONNXTensorElementDataType
CustomGemmOpFloat8E4M3FN::GetOutputType(size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
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
  beta_ = KernelInfoGetOptionalAttribute<float>(api, info, "beta", 0);

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

#if CUDA_VERSION >= 12000
  EXt_ENFORCE(beta_ == 0, "beta != 0 only supported for CUDA >= 12.0.");
#endif
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
  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
  CUDA_THROW_IF_ERROR(cudaStreamSynchronize(stream));
  auto time0 = std::chrono::high_resolution_clock::now();

  int n_inputs = ctx.GetInputCount();
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue input_C;
  bool has_bias;
  if (n_inputs > 2) {
    input_C = ctx.GetInput(2);
    has_bias = input_C.IsTensor();
  } else {
    has_bias = false;
  }
  Ort::ConstValue scale_A, scale_B, scale_Y;

  auto memA = input_A.GetTensorMemoryInfo();
  EXT_ENFORCE(memA.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input A is not on CUDA");
  auto memB = input_B.GetTensorMemoryInfo();
  EXT_ENFORCE(memB.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input B is not on CUDA");
  if (has_bias) {
    auto memC = input_C.GetTensorMemoryInfo();
    EXT_ENFORCE(memC.GetDeviceType() ==
                    OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                "Input C is not on CUDA");
  }

  bool has_scales = n_inputs == 6;
  if (has_scales) {
    scale_A = ctx.GetInput(3);
    scale_B = ctx.GetInput(4);
    scale_Y = ctx.GetInput(5);
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
  } else if (n_inputs != 2 && n_inputs != 3) {
    EXT_THROW("Number of inputs must be 2, 3 or 6 but is ", n_inputs, ".");
  }

  std::vector<int64_t> a_shape = input_A.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> b_shape = input_B.GetTensorTypeAndShapeInfo().GetShape();
  auto dtype_A = input_A.GetTensorTypeAndShapeInfo().GetElementType();
  auto dtype_B = input_B.GetTensorTypeAndShapeInfo().GetElementType();

  EXT_ENFORCE(a_shape.size() == 2);
  EXT_ENFORCE(b_shape.size() == 2);

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

  if (has_bias) {
    ONNXTensorElementDataType c_dtype =
        input_C.GetTensorTypeAndShapeInfo().GetElementType();
    EXT_ENFORCE(c_dtype == out_dtype,
                "dtype of input C and output dtype must be the same.");
  }

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
  if (has_scales) {
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
  size_t workspaceSize = (size_t)(1 << 25); // suggested fixed value 32Mb
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
      ", alpha=", alpha_, ", n_inputs=", n_inputs,
      ", A_type=", CudaDataTypeToString(a_cuda_type),
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
  const void *bias =
      has_bias ? input_C.GetTensorRawData() : Y.GetTensorMutableRawData();
  cuda_status = cublasLtMatmul(
      cublasLt, operationDesc, static_cast<const void *>(&alpha_), /* alpha */
      input_A.GetTensorRawData(),                                  /* A */
      Adesc, input_B.GetTensorRawData(),                           /* B */
      Bdesc, static_cast<const void *>(&beta_),                    /* beta */
      bias,                                                        /* C */
      Cdesc, Y.GetTensorMutableRawData(),                          /* Y */
      Ddesc, &heuristicResult.algo,                                /* algo */
      workspace,              /* workspace */
      workspaceSize, stream); /* stream */
  EXT_ENFORCE(
      cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to run cublasLtMatmul due to ", cublasGetErrorEnum(cuda_status),
      ", returnedResults=", returnedResults, ", alpha=", alpha_,
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
      ", rowMajor=", (row_major_ ? 1 : 0), ".");

  if (workspaceSize > 0) {
    CUDA_THROW_IF_ERROR(cudaFree(workspace));
  }

  CUBLAS_THROW_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Ddesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  CUBLAS_THROW_IF_ERROR(cublasLtDestroy(cublasLt));

  CUDA_THROW_IF_ERROR(cudaStreamSynchronize(stream));
  std::vector<int64_t> tdims{1};
  Ort::UnownedValue ttime = ctx.GetOutput(1, tdims);
  void *ptr_time = ttime.GetTensorMutableRawData();
  double performance = std::chrono::duration<double>(
                           std::chrono::high_resolution_clock::now() - time0)
                           .count();
  CUDA_THROW_IF_ERROR(cudaMemcpy(ptr_time, &performance, sizeof(double),
                                 cudaMemcpyHostToDevice));
}

} // namespace ortops
