#include "cuda_gemm.cuh"
#include "cuda_nvtx.cuh"
#include "cuda_utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#define NVTE_ERROR(x)                                                          \
  do {                                                                         \
    throw std::runtime_error(std::string(__FILE__ ":") +                       \
                             std::to_string(__LINE__) + " in function " +      \
                             __func__ + ": " + x);                             \
  } while (false)

#define NVTE_CHECK(x, ...)                                                     \
  do {                                                                         \
    if (!(x)) {                                                                \
      NVTE_ERROR(std::string("Assertion failed: " #x ". ") +                   \
                 std::string(__VA_ARGS__));                                    \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_CUDA(ans)                                                   \
  {                                                                            \
    auto status = ans;                                                         \
    NVTE_CHECK(status == cudaSuccess,                                          \
               "CUDA Error: " + std::string(cudaGetErrorString(status)));      \
  }

#define NVTE_CHECK_CUBLAS(ans)                                                 \
  {                                                                            \
    auto status = ans;                                                         \
    NVTE_CHECK(status == CUBLAS_STATUS_SUCCESS,                                \
               "CUBLAS Error: " + std::string(cublasGetStatusString(status))); \
  }

namespace cuda_example {

bool is_fp8_dtype(cudaDataType_t dtype) {
  switch (dtype) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
    return true;
  case CUDA_R_8F_E5M2:
    return true;
#endif
  default:
    return false;
  }
}

int32_t type_size(cudaDataType_t element_type) {
  switch (element_type) {
  case CUDA_R_32F:
    return 4;
  case CUDA_R_16F:
  case CUDA_R_16BF:
    return 2;
  case CUDA_R_8I:
  case CUDA_R_8U:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
  case CUDA_R_8F_E5M2:
#endif
    return 1;
  default:
    throw std::runtime_error("Unkown data type.");
  }
}

cudaDataType_t get_cuda_dtype(cudaDataType_t dtype) { return dtype; }

typedef enum TensorDevice { CPU = 0, CUDA = 1 } TensorDevice;

struct TensorData {
  TensorDevice device;
  cudaDataType_t dtype;
  size_t size;
  void *dptr;
  TensorData() {
    device = TensorDevice::CPU;
    size = 0;
    dptr = nullptr;
    dtype = CUDA_R_32F;
  }
  void allocate(cudaDataType_t dtype, size_t size, TensorDevice device) {
    this->dtype = dtype;
    this->size = size;
    this->device = device;
    switch (device) {
    case TensorDevice::CPU:
      dptr = malloc(size * type_size(dtype));
      std::cout << "CPUAllocate(" << size << " * " << type_size(dtype) << ")\n";
      break;
    case TensorDevice::CUDA:
      if (cudaMalloc(&dptr, size * type_size(dtype)) != cudaSuccess) {
        std::ostringstream st;
        st << "Unable to allocate " << size << " bytes on GPU.";
        NVTE_ERROR(std::string(st.str()));
      }
      std::cout << "CUDAAllocate(" << size << " * " << type_size(dtype) << ")\n";
      break;
    }
  }
  void free() {
    if (dptr != nullptr) {
      switch (device) {
      case TensorDevice::CPU:
        std::cout << "FreeCPU\n";
        ::free(dptr);
        break;
      case TensorDevice::CUDA:
        std::cout << "FreeCUDA\n";
        NVTE_CHECK_CUDA(cudaFree(dptr));
        break;
      }
      dptr = nullptr;
    }
  }
};

class Tensor {
public:
  const char *name;
  TensorData data;
  TensorData scale;
  TensorData amax;
  TensorData scale_inv;

public:
  Tensor(const char *name) { this->name = name; }
  Tensor(const char *name, size_t size, cudaDataType_t dtype = CUDA_R_32F,
         TensorDevice device = TensorDevice::CUDA) {
    this->name = name;
    std::cout << "T0:" << name << ":" << size << "\n";
    data.allocate(dtype, size, device);
    if (is_fp8_dtype(dtype)) {
      float one = 1;
      std::cout << "T1a\n";
      std::cout << "T1b\n";
      scale.allocate(CUDA_R_32F, 1, device);
      std::cout << "T1c\n";
      NVTE_CHECK_CUDA(
          cudaMemcpy(scale.dptr, &one, sizeof(float), cudaMemcpyHostToDevice));

      std::cout << "T2\n";
      scale_inv.allocate(CUDA_R_32F, 1, device);
      NVTE_CHECK_CUDA(cudaMemcpy(scale_inv.dptr, &one, sizeof(float),
                                 cudaMemcpyHostToDevice));
      std::cout << "T3\n";
    }
  }
  ~Tensor() {
    std::cout << "free1:" << name << "\n";
    data.free();
    std::cout << "free2\n";
    scale.free();
    std::cout << "free3\n";
    scale_inv.free();
    std::cout << "free4\n";
    amax.free();
    std::cout << "free5\n";
  }
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n,
                 int k, int lda, int ldb, int ldd, cublasOperation_t transa,
                 cublasOperation_t transb, bool grad, void *workspace,
                 size_t workspaceSize, bool accumulate,
                 bool use_split_accumulator, int math_sm_count,
                 cudaStream_t stream) {
  std::cout << "GEMM0\n";
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;
  std::cout << "GEMM0-1\n";
  void *C = outputD->data.dptr;
  void *D = outputD->data.dptr;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;
  std::cout << "GEMM0-2\n";
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  std::cout << "GEMM0-3\n";
  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 =
      is_fp8_dtype(inputA->data.dtype) || is_fp8_dtype(inputB->data.dtype);
  const cudaDataType_t A_type = get_cuda_dtype(inputA->data.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(inputB->data.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(outputD->data.dtype);
  const cudaDataType_t bias_type = get_cuda_dtype(inputBias->data.dtype);
  std::cout << "GEMM1\n";

  NVTE_CHECK(!is_fp8_dtype(inputA->data.dtype) || A_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(inputB->data.dtype) || B_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  std::cout << "GEMM2\n";
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

  std::cout << "GEMM3\n";
  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t)ldd;

  std::cout << "GEMM4\n";
  // default to tf32 except for e5m2 inputs where the config is not supported
  cublasComputeType_t gemm_compute_type =
      (A_type == CUDA_R_8F_E5M2 || B_type == CUDA_R_8F_E5M2)
          ? CUBLAS_COMPUTE_32F
          : CUBLAS_COMPUTE_32F_FAST_TF32;

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

  std::cout << "GEMM5\n";
  // set fp8 attributes -- input and output types should already be set to fp8
  // as appropriate Note: gelu fusion isn't available right now, and we don't
  // need amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    std::cout << "GEMM6\n";
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
      std::cout << "GEMM6-a\n";
      // Accumulation mode not supported for FP8 output
      C = nullptr;
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &D_scale,
          sizeof(D_scale)));
      std::cout << "GEMM6-c\n";
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax,
          sizeof(D_amax)));
      std::cout << "GEMM6-c\n";
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
    std::cout << "GEMM7\n";
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  }

  if (bias && gelu) {
    std::cout << "GEMM8\n";
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
    std::cout << "GEMM10\n";
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
    std::cout << "GEMM11\n";
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

  std::cout << "GEMM12\n";
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
      sizeof(epilogue)));

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
      sizeof(workspaceSize)));

  NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults));

  std::cout << "GEMM13\n";
  if (returnedResults == 0)
    throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C

  std::cout << "GEMM14\n";
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

  std::cout << "GEMM15\n";
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
  std::cout << "GEMM16\n";
}

#endif

void gemm_test(int test) {

  cudaDataType_t type_a, type_b, type_d;
  switch (test) {
  case 0:
    type_a = CUDA_R_32F;
    type_b = CUDA_R_32F;
    type_d = CUDA_R_32F;
    break;
  case 1:
    type_a = CUDA_R_8F_E4M3;
    type_b = CUDA_R_8F_E4M3;
    type_d = CUDA_R_16BF;
    break;
  default:
    throw std::runtime_error("Unknown test.");
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

  int dim = 16;

  std::cout << "GEMM0 - test=" << test << "\n";

  Tensor inputA("A", dim * dim, type_a);
  std::cout << "GEMM0-a" << test << "\n";
  Tensor inputB("B", dim * dim, type_b);
  Tensor outputD("D", dim * dim, type_d);
  Tensor workspace("workspace", 1 << 20, CUDA_R_8I);
  Tensor inputBias("bias");
  Tensor outputPreGelu("outputPreGelu");

  std::cout << "+stream\n";

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  std::cout << "cublas_gemm\n";

  cublas_gemm(&inputA, &inputB, &outputD, &inputBias, &outputPreGelu, dim, dim,
              dim,                 // int m, int n, int k,
              dim, dim, dim,       // int lda, int ldb, int ldd,
              CUBLAS_OP_T,         // cublasOperation_t transa,
              CUBLAS_OP_N,         // cublasOperation_t transb,
              false,               // bool grad,
              workspace.data.dptr, // void* workspace,
              workspace.data.size, // size_t workspaceSize,
              false,               // bool accumulate,
              false,               // bool use_split_accumulator,
              0,                   // int math_sm_count,
              stream);

  std::cout << "cublas_gemm done\n";

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));

  std::cout << "end\n";

#else

  throw std::runtime_error("Test not available for CUDA_VERSION < 11.8.");

#endif
}

} // namespace cuda_example
