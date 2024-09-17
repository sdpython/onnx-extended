#include "cuda/common_kernels_cuda.h"
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

//////////////////
// CustomGemmOp...
//////////////////

void *CustomGemmOp::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char *CustomGemmOp::GetName() const { return op_name_; }

const char *CustomGemmOp::GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

size_t CustomGemmOp::GetInputTypeCount() const { return 6; };

ONNXTensorElementDataType CustomGemmOp::GetInputType(std::size_t index) const {
  switch (index) {
  case 0: // A
  case 1: // B
    return ab_type_;
  case 2: // C
    return c_type_;
  case 3: // scale A
  case 4: // scale B
  case 5: // scale Y
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    EXT_THROW("Input index=", (int64_t)index, " is out of boundary.");
  }
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOp::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
  case 1:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 2:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  case 3:
  case 4:
  case 5:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

size_t CustomGemmOp::GetOutputTypeCount() const { return compute_time_as_output_ ? 2 : 1; }

ONNXTensorElementDataType CustomGemmOp::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return d_type_;
  case 1:
    if (!compute_time_as_output_) {
      EXT_THROW("Output index=", (uint64_t)index,
                " is out of boundary, compute_time_as_output_ is False.");
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOp::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  case 1:
    if (!compute_time_as_output_) {
      EXT_THROW("Output index=", (uint64_t)index,
                " is out of boundary, compute_time_as_output_ is False.");
    }
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// CustomGemmKernel
///////////////////

CustomGemmKernel::CustomGemmKernel(const OrtApi &api, const OrtKernelInfo *info) {
  rowMajor_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "rowMajor", 1);
  transA_ = KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transA", false);
  transB_ = KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transB", false);
  fastAccumulationMode_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "fastAccumulationMode", true);
  smCount_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "smCount", 0);
  alpha_ = KernelInfoGetOptionalAttribute<float>(api, info, "alpha", 1);
  beta_ = KernelInfoGetOptionalAttribute<float>(api, info, "beta", 0);

  // A string attribute.
  std::string compute_type =
      KernelInfoGetOptionalAttributeString(api, info, "computeType", "CUBLAS_COMPUTE_32F");
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
  } else if (compute_type == "CUBLAS_COMPUTE_32I") {
    computeType_ = CUBLAS_COMPUTE_32I;
  } else {
    EXT_THROW("Unexpected value for compute_type '", compute_type, "'.");
  }

  std::string activation =
      KernelInfoGetOptionalAttributeString(api, info, "activation", "DEFUALT");
  if (activation == "DEFUALT") {
    epilogue_ = EpiloqueGemmKernel::Default;
  } else if (activation == "RELU") {
    epilogue_ = EpiloqueGemmKernel::Relu;
  } else if (activation == "GELU") {
    epilogue_ = EpiloqueGemmKernel::Gelu;
  } else {
    EXT_THROW("Unexpected value for activation '", activation, "'.");
  }

#if CUDA_VERSION < 12000
  EXT_ENFORCE(beta_ == 0, "beta != 0 only supported for CUDA >= 12.0.");
#endif
}

void CustomGemmKernel::SetParams(const std::vector<int64_t> &shape_A,
                                 const std::vector<int64_t> &shape_B, int &M, int &N, int &K,
                                 int &lda, int &ldb, int &ldd, int row_major) const {
  constexpr int ir = 0;
  constexpr int ic = 1 - ir;
  lda = shape_A[row_major ? ic : ir];
  ldb = shape_B[row_major ? ic : ir];
  if (transB_) {
    if (transA_) { // NT
      M = shape_A[ic];
      N = shape_B[ir];
      K = shape_A[ir];
    } else { // TT
      M = shape_A[ir];
      N = shape_B[ir];
      K = shape_A[ic];
    }
    ldd = shape_B[row_major ? ir : ic];
  } else {
    if (transA_) { // TN
      M = shape_A[ic];
      N = shape_B[ic];
      K = shape_A[ir];
    } else { // NN
      M = shape_A[ir];
      N = shape_B[ic];
      K = shape_A[ic];
    }
    ldd = shape_B[row_major ? ic : ir];
  }
}

static void check_device(const Ort::ConstValue &input, const char *name) {
  EXT_ENFORCE(input.HasValue(), "Input '", name, "' is not empty.");
  auto mem = input.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Input '", name, "' is not on CUDA");
}

static void check_device(const Ort::UnownedValue &output, const char *name) {
  auto mem = output.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
              "Output '", name, "' is not on CUDA");
}

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue &input, std::vector<int64_t> &shape,
                                          bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();
  EXT_ENFORCE(shape.size() == 2);
  if (swap) {
    std::swap(shape[0], shape[1]);
  }
  return t.GetElementType();
}

void CustomGemmKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  Ort::ConstValue scale_A, scale_B, scale_Y;
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue input_C;
  bool has_bias;
  if (n_inputs > 2) {
    input_C = ctx.GetInput(2);
    has_bias = beta_ != 0 && input_C.HasValue() && input_C.IsTensor();
  } else {
    has_bias = false;
  }

  check_device(input_A, "A");
  check_device(input_B, "B");
  if (has_bias)
    check_device(input_C, "C");

  bool has_scales = n_inputs > 3;
  bool has_scales_Y = n_inputs > 5;
  if (has_scales) {
    EXT_ENFORCE(n_inputs == 5 || n_inputs == 6, "Number of inputs must be 5 or 6 but is ",
                (int64_t)n_inputs, ".");
    scale_A = ctx.GetInput(3);
    scale_B = ctx.GetInput(4);
    check_device(scale_A, "scale_A");
    check_device(scale_B, "scale_B");
    if (has_scales_Y) {
      scale_Y = ctx.GetInput(5);
      check_device(scale_Y, "scale_Y");
    }
  } else if (n_inputs != 2 && n_inputs != 3) {
    EXT_THROW("Number of inputs must be 2, 3 or 6 but is ", (int64_t)n_inputs, ".");
  }

  switch (rowMajor_) {
  case 0:
    ComputeColMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A, input_B,
                    input_C, scale_A, scale_B, scale_Y);
    break;
  case 1:
    ComputeRowMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A, input_B,
                    input_C, scale_A, scale_B, scale_Y);
    break;
  default:
    EXT_THROW("Unexpected value for rowMajor_=", rowMajor_, ".");
  }
}

void CustomGemmKernel::ComputeRowMajor(Ort::KernelContext &ctx, int n_inputs, bool has_bias,
                                       bool has_scales, bool has_scales_Y,
                                       Ort::ConstValue &input_A, Ort::ConstValue &input_B,
                                       Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
                                       Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  SetParams(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C = has_bias ? GetTypeAndShape(input_C, shape_C) : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_A, dtype_B, dtype_C,
              dtype_Y, shape_A, shape_B, shape_C, shape_Y, transA_, transB_,
              input_A.GetTensorRawData(), input_B.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr, Y.GetTensorMutableRawData(),
              M, N, K, lda, ldb, ldd);
}

void CustomGemmKernel::ComputeColMajor(Ort::KernelContext &ctx, int n_inputs, bool has_bias,
                                       bool has_scales, bool has_scales_Y,
                                       Ort::ConstValue &input_A, Ort::ConstValue &input_B,
                                       Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
                                       Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  SetParams(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::swap(shape_A[0], shape_A[1]);
  std::swap(shape_B[0], shape_B[1]);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C =
      has_bias ? GetTypeAndShape(input_C, shape_C, true) : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_B, dtype_A, dtype_C,
              dtype_Y, shape_B, shape_A, shape_C, shape_Y, transB_, transA_,
              input_B.GetTensorRawData(), input_A.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr, Y.GetTensorMutableRawData(),
              N, M, K, ldb, lda, ldd);
}

void CustomGemmKernel::ComputeGemm(
    Ort::KernelContext &ctx, int n_inputs, bool has_bias, bool has_scales, bool has_scales_Y,
    ONNXTensorElementDataType dtype_A, ONNXTensorElementDataType dtype_B,
    ONNXTensorElementDataType dtype_C, ONNXTensorElementDataType dtype_Y,
    const std::vector<int64_t> &shape_A, const std::vector<int64_t> &shape_B,
    const std::vector<int64_t> &shape_C, const std::vector<int64_t> &shape_Y, bool trans_A,
    bool trans_B, const void *p_input_a, const void *p_input_b, const void *p_input_c,
    const void *p_scale_a, const void *p_scale_b, const void *p_scale_y, void *p_output_y,
    int M, int N, int K, int lda, int ldb, int ldd) {
  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
  CUDA_THROW_IF_ERROR(cudaStreamSynchronize(stream));
  auto time0 = std::chrono::high_resolution_clock::now();

  cublasLtHandle_t cublasLt;
  CUBLAS_THROW_IF_ERROR(cublasLtCreate(&cublasLt));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;

  // Create matrix descriptors. Not setting any extra attributes.
  cudaDataType_t a_cuda_type = ToCudaDataType(dtype_A);
  cudaDataType_t b_cuda_type = ToCudaDataType(dtype_B);
  cudaDataType_t d_cuda_type = ToCudaDataType(dtype_Y);
  cudaDataType_t scale_cuda_type = ToCudaDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  cudaDataType_t bias_cuda_type = ToCudaDataType(dtype_C);

  CUBLAS_THROW_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Adesc, a_cuda_type, trans_A ? K : M, trans_A ? M : K, lda));
  CUBLAS_THROW_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Bdesc, b_cuda_type, trans_B ? N : K, trans_B ? K : N, ldb));
  CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, d_cuda_type, M, N, ldd));

  if (rowMajor_) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                           &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                           &matrixOrder, sizeof(matrixOrder)));
  }

  CUBLAS_THROW_IF_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, computeType_, scale_cuda_type));
  cublasOperation_t ctransa = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t ctransb = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &ctransa, sizeof(ctransa)));
  CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &ctransb, sizeof(ctransb)));

  if (smCount_ != 0) {
    int math_sm_count = static_cast<int>(smCount_);
    CUBLAS_THROW_IF_ERROR(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                       &math_sm_count, sizeof(math_sm_count)));
  }

  if (has_scales) {
    // gemm float 8
    const int8_t ifast_accumulation_mode = fastAccumulationMode_ ? 1 : 0;
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &ifast_accumulation_mode, sizeof(ifast_accumulation_mode)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p_scale_a, sizeof(p_scale_a)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p_scale_b, sizeof(p_scale_b)));
    if (has_scales_Y) {
      CUBLAS_THROW_IF_ERROR(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &p_scale_y, sizeof(p_scale_b)));
    }

    // float 8
#if ORT_VERSION >= 1160 && CUDA_VERSION >= 11080
    if (dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
        dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2) {
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, bias_cuda_type, M, N, ldd));
      CUBLAS_THROW_IF_ERROR(
          cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                         &bias_cuda_type, sizeof(bias_cuda_type)));
    } else {
      CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
    }
#endif
  } else {
    CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, d_cuda_type, M, N, ldd));
  }

  if (rowMajor_) {
    cublasLtOrder_t matrixOrder = CUBLASLT_ORDER_ROW;
    CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                           &matrixOrder, sizeof(matrixOrder)));
    CUBLAS_THROW_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                           &matrixOrder, sizeof(matrixOrder)));
  }

  cublasLtEpilogue_t epilogue;
  switch (epilogue_) {
  case EpiloqueGemmKernel::Default:
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    break;
  case EpiloqueGemmKernel::Relu:
    epilogue = CUBLASLT_EPILOGUE_RELU;
    break;
  case EpiloqueGemmKernel::Gelu:
    epilogue = CUBLASLT_EPILOGUE_GELU;
    break;
  }
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                 sizeof(epilogue));

  // See
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true
  // with H100).
  std::size_t workspaceSize = (std::size_t)(1 << 25); // suggested fixed value 32Mb
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspaceSize, sizeof(workspaceSize));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResults = 0;
  cublasStatus_t cuda_status =
      cublasLtMatmulAlgoGetHeuristic(cublasLt, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                     preference, 1, &heuristicResult, &returnedResults);
  EXT_ENFORCE(returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
              " Unable to find any suitable algorithm due to ", cublasGetErrorEnum(cuda_status),
              ", returnedResults=", returnedResults, ", alpha=", alpha_, ", beta=", beta_,
              ", n_inputs=", n_inputs, ", A_type=", CudaDataTypeToString(a_cuda_type),
              ", B_type=", CudaDataTypeToString(b_cuda_type),
              ", C_type=", CudaDataTypeToString(bias_cuda_type),
              ", result_type=", CudaDataTypeToString(d_cuda_type),
              ", bias_type=", CudaDataTypeToString(bias_cuda_type),
              ", scale_type=", CudaDataTypeToString(scale_cuda_type),
              ", computeType=", CublasComputeTypeToString(computeType_),
              ", epilogue=", epilogue, ", smCount=", smCount_, ", transA=", trans_A,
              ", transB=", trans_B, ", fastAccumulationMode=", (fastAccumulationMode_ ? 1 : 0),
              ", shape_A=", shape_A[0], "x", shape_A[1], ", shape_B=", shape_B[0], "x",
              shape_B[1], ", shape_C=", (shape_C.size() > 0 ? shape_C[0] : 0), "x",
              (shape_C.size() > 1 ? shape_C[1] : 0), ", M=", M, ", N=", N, ", K=", K,
              ", lda=", lda, ", ldb=", ldb, ", ldd=", ldd, ", workspaceSize=", workspaceSize,
              ", rowMajor=", (rowMajor_ ? 1 : 0),
              ". Check NVIDIA documentation to see what combination is valid: ",
              "https://docs.nvidia.com/cuda/cublas/"
              "index.html?highlight=cublasLtMatmulAlgoGetHeuristic#"
              "cublasltmatmulalgogetheuristic.");

  void *workspace = nullptr;
  if (workspaceSize > 0) {
    CUDA_THROW_IF_ERROR(cudaMalloc((void **)&workspace, workspaceSize));
  }
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  const void *bias = has_bias ? p_input_c : p_output_y;
  cuda_status =
      cublasLtMatmul(cublasLt, operationDesc, static_cast<const void *>(&alpha_), /* alpha */
                     p_input_a,                                                   /* A */
                     Adesc, p_input_b,                                            /* B */
                     Bdesc, static_cast<const void *>(&beta_),                    /* beta */
                     bias,                                                        /* C */
                     Cdesc, p_output_y,                                           /* Y */
                     Ddesc, &heuristicResult.algo,                                /* algo */
                     workspace,              /* workspace */
                     workspaceSize, stream); /* stream */
  EXT_ENFORCE(cuda_status == CUBLAS_STATUS_SUCCESS, " Unable to run cublasLtMatmul due to ",
              cublasGetErrorEnum(cuda_status), ", returnedResults=", returnedResults,
              ", alpha=", alpha_, ", n_inputs=", n_inputs,
              ", A_type=", CudaDataTypeToString(a_cuda_type),
              ", B_type=", CudaDataTypeToString(b_cuda_type),
              ", result_type=", CudaDataTypeToString(d_cuda_type),
              ", bias_type=", CudaDataTypeToString(bias_cuda_type),
              ", scale_type=", CudaDataTypeToString(scale_cuda_type),
              ", computeType=", CublasComputeTypeToString(computeType_),
              ", epilogue=", epilogue, ", smCount=", smCount_, ", transA=", trans_A,
              ", transB=", trans_B, ", fastAccumulationMode=", (fastAccumulationMode_ ? 1 : 0),
              ", shape_A=", shape_A[0], "x", shape_A[1], ", shape_B=", shape_B[0], "x",
              shape_B[1], ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb,
              ", ldd=", ldd, ", workspaceSize=", workspaceSize,
              ", rowMajor=", (rowMajor_ ? 1 : 0), ".");

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

  int n_outputs = ctx.GetOutputCount();
  if (n_outputs >= 2) {
    CUDA_THROW_IF_ERROR(cudaStreamSynchronize(stream));
    std::vector<int64_t> tdims{1};
    Ort::UnownedValue ttime = ctx.GetOutput(1, tdims);
    void *ptr_time = ttime.GetTensorMutableRawData();
    double performance =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0)
            .count();
    CUDA_THROW_IF_ERROR(
        cudaMemcpy(ptr_time, &performance, sizeof(double), cudaMemcpyHostToDevice));
  }
}

} // namespace ortops
