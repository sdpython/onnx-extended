#include "cuda_gemm.cuh"
#include "cuda_nvtx.cuh"
#include "cuda_tensor.cuh"
#include "cuda_utils.h"
#include <chrono>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_type;

namespace cuda_example {

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n,
                 int k, int lda, int ldb, int ldd, cublasOperation_t transa,
                 cublasOperation_t transb, bool grad, void *workspace,
                 std::size_t workspaceSize, bool accumulate,
                 bool use_split_accumulator, int math_sm_count,
                 cublasComputeType_t gemm_compute_type, cudaStream_t stream,
                 time_type &begin, time_type &heuristic, time_type &end,
                 time_type &end2, int &i_epilogue, int &i_compute_type,
                 int &i_algo) {
  begin = std::chrono::high_resolution_clock::now();
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;

  void *C = outputD->data.dptr;
  void *D = outputD->data.dptr;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;

  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;

  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 =
      is_fp8_dtype(inputA->data.dtype) || is_fp8_dtype(inputB->data.dtype);
  const cudaDataType_t A_type = get_cuda_dtype(inputA->data.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(inputB->data.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(outputD->data.dtype);
  const cudaDataType_t bias_type = get_cuda_dtype(inputBias->data.dtype);

  NVTE_CHECK(!is_fp8_dtype(inputA->data.dtype) || A_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(inputB->data.dtype) || B_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8 && gelu) {
    NVTE_CHECK(!is_fp8_dtype(outputPreGelu->data.dtype),
               "fp8 Aux output for gemm + gelu fusion not supported!");
  }
  if (is_fp8_dtype(outputD->data.dtype)) {
    NVTE_CHECK(!accumulate,
               "Accumulation mode not supported with FP8 GEMM output!");
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t)ldd;

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_CUBLAS(
      cublasLtMatrixLayoutCreate(&Adesc, A_type, transa == CUBLAS_OP_N ? m : k,
                                 transa == CUBLAS_OP_N ? k : m, lda));
  NVTE_CHECK_CUBLAS(
      cublasLtMatrixLayoutCreate(&Bdesc, B_type, transb == CUBLAS_OP_N ? k : n,
                                 transb == CUBLAS_OP_N ? n : k, ldb));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  // Set math SM count
  if (math_sm_count != 0) {
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count,
        sizeof(math_sm_count)));
  }

  // set fp8 attributes -- input and output types should already be set to fp8
  // as appropriate Note: gelu fusion isn't available right now, and we don't
  // need amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode,
        sizeof(fastAccuMode)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scale_inverse,
        sizeof(A_scale_inverse)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scale_inverse,
        sizeof(B_scale_inverse)));
    if (is_fp8_dtype(outputD->data.dtype)) {
      // Accumulation mode not supported for FP8 output
      C = nullptr;
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &D_scale,
          sizeof(D_scale)));
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax,
          sizeof(D_amax)));
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      NVTE_CHECK_CUBLAS(
          cublasLtMatrixLayoutCreate(&Cdesc, bias_type, m, n, ldd));
    } else {
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
    }
    if (bias) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type)));
    }
  } else {
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu_out,
        sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat,
        sizeof(ld_gelumat)));
    const cudaDataType_t aux_type = get_cuda_dtype(outputPreGelu->data.dtype);
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type,
        sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu_out,
        sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat,
        sizeof(ld_gelumat)));
  }
  i_epilogue = (int)epilogue;

  cublasLtMatmulPreference_t preference = nullptr;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
      sizeof(epilogue)));

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
      sizeof(workspaceSize)));

  NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults));

  if (returnedResults == 0)
    NVTE_ERROR("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C
  std::size_t written;
  NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
      &heuristicResult.algo, CUBLASLT_ALGO_CONFIG_ID, &i_algo, sizeof(int),
      &written));

  i_compute_type = (int)gemm_compute_type;

  heuristic = std::chrono::high_resolution_clock::now();

  NVTE_CHECK_CUBLAS(cublasLtMatmul(
      handle, operationDesc, static_cast<const void *>(&one), /* alpha */
      A,                                                      /* A */
      Adesc, B,                                               /* B */
      Bdesc, static_cast<const void *>(&beta),                /* beta */
      C,                                                      /* C */
      Cdesc, D,                                               /* D */
      Ddesc, &heuristicResult.algo,                           /* algo */
      workspace, workspaceSize,                               /* workspace */
      stream));                                               /* stream */

  end = std::chrono::high_resolution_clock::now();

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

  end2 = std::chrono::high_resolution_clock::now();
}

double to_double(const time_type &a, const time_type &b) {
  return std::chrono::duration<double>(b - a).count();
}

BenchmarkGemm::BenchmarkGemm() { zero(); }

void BenchmarkGemm::zero() {
  N = 0;
  workspace_new = 0;
  workspace_free = 0;
  stream_create = 0;
  stream_destroy = 0;
  setup = 0;
  gemm = 0;
  gemm_in = 0;
  gemm_sync = 0;
  clean = 0;
  total = 0;
}

void BenchmarkGemm::to_map(std::unordered_map<std::string, double> &bench) {
  bench["N"] = N;
  bench["t-workspace_new"] = workspace_new;
  bench["t-workspace_free"] = workspace_free;
  bench["t-stream_create"] = stream_create;
  bench["t-stream_destroy"] = stream_destroy;
  bench["t-setup"] = setup;
  bench["t-gemm_in"] = gemm_in;
  bench["t-gemm"] = gemm;
  bench["t-gemm_sync"] = gemm_sync;
  bench["t-clean"] = clean;
  bench["t-total"] = total;
}

std::unordered_map<std::string, double> gemm_benchmark_test(int test, int N,
                                                            int m, int n, int k,
                                                            int lda, int ldb,
                                                            int ldd) {

  // see
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatMul#cublasltmatmul
  cudaDataType_t type_a, type_b, type_d;
  cublasComputeType_t type_compute;
  switch (test) {
  case 0:
    type_a = CUDA_R_32F;
    type_b = CUDA_R_32F;
    type_d = CUDA_R_32F;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 1:
    type_a = CUDA_R_32F;
    type_b = CUDA_R_32F;
    type_d = CUDA_R_32F;
    type_compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    break;
  case 2:
    type_a = CUDA_R_32F;
    type_b = CUDA_R_32F;
    type_d = CUDA_R_32F;
    type_compute = CUBLAS_COMPUTE_32F_FAST_16BF;
    break;
  case 3:
    type_a = CUDA_R_16F;
    type_b = CUDA_R_16F;
    type_d = CUDA_R_16F;
    type_compute = CUBLAS_COMPUTE_16F;
    break;
  case 4:
    type_a = CUDA_R_16BF;
    type_b = CUDA_R_16BF;
    type_d = CUDA_R_16BF;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case 5:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_16BF;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 6:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_16BF;
    // default to tf32 except for e5m2 inputs where the config is not supported
    type_compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    break;
  case 7:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_32F;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 8:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_8F_E4M3;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 9:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E5M2;
    type_d = CUDA_R_8F_E4M3;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 10:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E5M2;
    type_d = CUDA_R_8F_E5M2;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 11:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E5M2;
    type_d = CUDA_R_16BF;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 12:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_8F_E4M3;
    type_compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    break;
  case 13:
    type_a = CUDA_R_8F_E5M2;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_8F_E5M2;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
  case 14:
    type_a = CUDA_R_8F_E5M2;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_16F;
    type_compute = CUBLAS_COMPUTE_32F;
    break;
#endif
  case 15:
    type_a = CUDA_R_8I;
    type_b = CUDA_R_8I;
    type_d = CUDA_R_32I;
    type_compute = CUBLAS_COMPUTE_32I;
    break;
  case 16:
    type_a = CUDA_R_8I;
    type_b = CUDA_R_8I;
    type_d = CUDA_R_8I;
    type_compute = CUBLAS_COMPUTE_32I;
    break;
  default:
    NVTE_CHECK(false, onnx_extended_helpers::MakeString(
                          "Unknown test ", test, " and this CUDA version ",
                          CUDA_VERSION, "."));
  }

  time_type begin, heuristic, end, end2;
  int epilogue, compute_type, algo;
  Tensor inputA("A", m * k, type_a);
  inputA.rnd();
  Tensor inputB("B", n * k, type_b);
  inputB.rnd();
  Tensor outputD("D", m * n, type_d);
  if (is_fp8_dtype(type_a) || is_fp8_dtype(type_b))
    outputD.amax.allocate(CUDA_R_32F, 1, outputD.scale.device);
  Tensor inputBias("bias");
  Tensor outputPreGelu("outputPreGelu");
  std::size_t workspace_size = 1 << 20;
  BenchmarkGemm results;

  cudaStream_t stream;

  for (int64_t i = 0; i < N; ++i) {
    time_type t0 = std::chrono::high_resolution_clock::now();
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));
    time_type t1 = std::chrono::high_resolution_clock::now();
    time_type t5;
    {
      Tensor workspace("workspace", 1 << 20, CUDA_R_8I);
      time_type t2 = std::chrono::high_resolution_clock::now();

      cublas_gemm(&inputA, &inputB, &outputD, &inputBias, &outputPreGelu, m, n,
                  k,                   // int m, int n, int k,
                  lda, ldb, ldd,       // int lda, int ldb, int ldd,
                  CUBLAS_OP_T,         // cublasOperation_t transa,
                  CUBLAS_OP_N,         // cublasOperation_t transb,
                  false,               // bool grad,
                  workspace.data.dptr, // void* workspace,
                  workspace.data.size, // std::size_t workspaceSize,
                  false,               // bool accumulate,
                  false,               // bool use_split_accumulator,
                  0,                   // int math_sm_count,
                  type_compute,        // compute_type
                  stream, begin, heuristic, end, end2, epilogue, compute_type,
                  algo);

      time_type t3 = std::chrono::high_resolution_clock::now();
      NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
      time_type t4 = std::chrono::high_resolution_clock::now();
      NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
      t5 = std::chrono::high_resolution_clock::now();
      results.workspace_new += to_double(t1, t2);
      results.gemm += to_double(t2, t3);
      results.gemm_sync += to_double(t2, t4);
      results.gemm_in += to_double(heuristic, end);
      results.stream_destroy += to_double(t4, t5);
      results.setup += to_double(begin, heuristic);
      results.clean += to_double(end, end2);
    }
    time_type t6 = std::chrono::high_resolution_clock::now();
    ++results.N;
    results.workspace_free += to_double(t5, t6);
    results.total += to_double(t0, t6);
  }

  std::unordered_map<std::string, double> bench;
  bench["epiloque"] = epilogue;
  bench["algo"] = algo;
  bench["compute_type"] = compute_type;
  bench["workspace_size"] = workspace_size;
  bench["m"] = m;
  bench["n"] = n;
  bench["k"] = k;
  bench["lda"] = lda;
  bench["ldb"] = ldb;
  bench["ldd"] = ldd;
  bench["type_a"] = type_a;
  bench["type_b"] = type_b;
  bench["type_d"] = type_d;
  results.to_map(bench);
  return bench;
}

} // namespace cuda_example
