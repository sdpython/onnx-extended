#include "custom_gemm.h"
#include "cpu/cast_fp8.h"
#include <omp.h>

namespace ortops {

//////////////////
// CustomGemmOp...
//////////////////

void *CustomGemmOp::CreateKernel(const OrtApi &api,
                                 const OrtKernelInfo *info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char *CustomGemmOp::GetName() const { return op_name_; }

const char *CustomGemmOp::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

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

size_t CustomGemmOp::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomGemmOp::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
  case 0:
    return d_type_;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOp::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
  case 0:
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  default:
    EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
  }
}

///////////////////
// CustomGemmKernel
///////////////////

CustomGemmKernel::CustomGemmKernel(const OrtApi &api,
                                   const OrtKernelInfo *info) {
  rowMajor_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "rowMajor", 1);
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
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (compute_type == "CUBLAS_COMPUTE_32F") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16F") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_TF32") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (compute_type == "CUBLAS_COMPUTE_32I") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
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

  ThrowOnError(api, api.KernelInfo_GetInputCount(info, &n_inputs_));
  if (n_inputs_ > 5) {
    std::string name = KernelInfoGetInputName(api, info, 5);
    has_scale_Y_ = !name.empty();
  } else {
    has_scale_Y_ = false;
  }
}

void CustomGemmKernel::set(const std::vector<int64_t> &shape_A,
                           const std::vector<int64_t> &shape_B, int &M, int &N,
                           int &K, int &lda, int &ldb, int &ldd,
                           int row_major) const {
  constexpr int ir = 0;
  constexpr int ic = 1 - ir;
  if (transA_ && !transB_) { // TN
    M = shape_A[ic];
    N = shape_B[ic];
    K = shape_A[ir];
    lda = shape_A[row_major ? ic : ir];
    ldb = shape_B[row_major ? ic : ir];
    ldd = shape_B[row_major ? ic : ir];
  } else if (!transA_ && !transB_) { // NN
    M = shape_A[ir];
    N = shape_B[ic];
    K = shape_A[ic];
    lda = shape_A[row_major ? ic : ir];
    ldb = shape_B[row_major ? ic : ir];
    ldd = shape_B[row_major ? ic : ir];
  } else if (!transA_ && transB_) { // NT
    M = shape_A[ir];
    N = shape_B[ir];
    K = shape_A[ic];
    lda = shape_A[row_major ? ic : ir];
    ldb = shape_B[row_major ? ic : ir];
    ldd = shape_B[row_major ? ir : ic];
  } else { // TT
    M = shape_A[ic];
    N = shape_B[ir];
    K = shape_A[ir];
    lda = shape_A[row_major ? ic : ir];
    ldb = shape_B[row_major ? ic : ir];
    ldd = shape_B[row_major ? ir : ic];
  }
}

void check_device(const Ort::ConstValue &input, const char *name) {
  EXT_ENFORCE(input.HasValue(), "Input '", name, "' is empty.");
  auto mem = input.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
              "Input '", name, "' is not on CPU");
}

void check_device(const Ort::UnownedValue &output, const char *name) {
  auto mem = output.GetTensorMemoryInfo();
  EXT_ENFORCE(mem.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
              "Output '", name, "' is not on CPU");
}

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue &input,
                                          std::vector<int64_t> &shape,
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

  int n_inputs = static_cast<int>(n_inputs_);
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
  bool has_scales_Y = n_inputs > 5 && has_scale_Y_;
  if (has_scales) {
    EXT_ENFORCE(n_inputs == 5 || n_inputs == 6,
                "Number of inputs must be 5 or 6 but is ", n_inputs, ".");
    scale_A = ctx.GetInput(3);
    scale_B = ctx.GetInput(4);
    check_device(scale_A, "scale_A");
    check_device(scale_B, "scale_B");
    if (has_scales_Y) {
      scale_Y = ctx.GetInput(5);
      check_device(scale_Y, "scale_Y");
    }
  } else if (n_inputs != 2 && n_inputs != 3) {
    EXT_THROW("Number of inputs must be 2, 3 or 6 but is ",
              (int64_t)n_inputs, ".");
  }

  switch (rowMajor_) {
  case 0:
    ComputeColMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A,
                    input_B, input_C, scale_A, scale_B, scale_Y);
    break;
  case 1:
    ComputeRowMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A,
                    input_B, input_C, scale_A, scale_B, scale_Y);
    break;
  default:
    EXT_THROW("Unexpected value for rowMajor_=", rowMajor_, ".");
  }
}

void CustomGemmKernel::ComputeRowMajor(
    Ort::KernelContext &ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, Ort::ConstValue &input_A, Ort::ConstValue &input_B,
    Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
    Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  set(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C = has_bias ? GetTypeAndShape(input_C, shape_C)
                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_A,
              dtype_B, dtype_C, dtype_Y, shape_A, shape_B, shape_C, shape_Y,
              transA_, transB_, input_A.GetTensorRawData(),
              input_B.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr,
              Y.GetTensorMutableRawData(), M, N, K, lda, ldb, ldd);
}

void CustomGemmKernel::ComputeColMajor(
    Ort::KernelContext &ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, Ort::ConstValue &input_A, Ort::ConstValue &input_B,
    Ort::ConstValue &input_C, Ort::ConstValue &scale_A,
    Ort::ConstValue &scale_B, Ort::ConstValue &scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  set(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::swap(shape_A[0], shape_A[1]);
  std::swap(shape_B[0], shape_B[1]);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C = has_bias ? GetTypeAndShape(input_C, shape_C, true)
                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_B,
              dtype_A, dtype_C, dtype_Y, shape_B, shape_A, shape_C, shape_Y,
              transA_, transB_, input_B.GetTensorRawData(),
              input_A.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr,
              Y.GetTensorMutableRawData(), N, M, K, ldb, lda, ldd);
}

void CustomGemmKernel::ComputeGemm(
    Ort::KernelContext &ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, ONNXTensorElementDataType dtype_A,
    ONNXTensorElementDataType dtype_B, ONNXTensorElementDataType dtype_C,
    ONNXTensorElementDataType dtype_Y, const std::vector<int64_t> &shape_A,
    const std::vector<int64_t> &shape_B, const std::vector<int64_t> &shape_C,
    const std::vector<int64_t> &shape_Y, bool trans_A, bool trans_B,
    const void *p_input_a, const void *p_input_b, const void *p_input_c,
    const void *p_scale_a, const void *p_scale_b, const void *p_scale_y,
    void *p_output_y, int M, int N, int K, int lda, int ldb, int ldd) {

  if (dtype_A == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_B == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_C == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      computeType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, shape_A,
                shape_B, shape_C, shape_Y, trans_A, trans_B,
                static_cast<const float *>(p_input_a),
                static_cast<const float *>(p_input_b),
                static_cast<const float *>(p_input_c),
                static_cast<const float *>(p_scale_a),
                static_cast<const float *>(p_scale_b),
                static_cast<const float *>(p_scale_y),
                static_cast<float *>(p_output_y), M, N, K, lda, ldb, ldd);
  } else if (dtype_A == 17 /* ONNX_TENSOR_ELEMENT_DATA_TYPE_E4M3FN */ &&
             dtype_B == 17 /* ONNX_TENSOR_ELEMENT_DATA_TYPE_E4M3FN */ &&
             dtype_C == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
             dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
             computeType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> c_input_a(M * K);
    std::vector<float> c_input_b(N * K);
    e4m3fn_to_float(c_input_a.size(), static_cast<const uint8_t *>(p_input_a),
                    c_input_a.data(), *(static_cast<const float *>(p_scale_a)));
    e4m3fn_to_float(c_input_b.size(), static_cast<const uint8_t *>(p_input_b),
                    c_input_b.data(), *(static_cast<const float *>(p_scale_b)));
    ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, shape_A,
                shape_B, shape_C, shape_Y, trans_A, trans_B, c_input_a.data(),
                c_input_b.data(), static_cast<const float *>(p_input_c),
                static_cast<const float *>(p_scale_a),
                static_cast<const float *>(p_scale_b),
                static_cast<const float *>(p_scale_y),
                static_cast<float *>(p_output_y), M, N, K, lda, ldb, ldd);
  } else {
    EXT_THROW("Not implemented for dtype_A=", (int64_t)dtype_A,
              " dtype_B=", (int64_t)dtype_B,
              " dtype_C=", (int64_t)dtype_C,
              " dtype_Y=", (int64_t)dtype_Y, ".");
  }
}

void CustomGemmKernel::ComputeGemm(
    Ort::KernelContext & /* ctx */, int /* n_inputs */, bool /* has_bias */, bool has_scales,
    bool has_scales_Y, const std::vector<int64_t> & /* shape_A */,
    const std::vector<int64_t> & /* shape_B */, const std::vector<int64_t> & /* shape_C */,
    const std::vector<int64_t> & /* shape_Y */, bool transa, bool transb,
    const float *p_input_a, const float *p_input_b, const float *p_input_c,
    const float *p_scale_a, const float *p_scale_b, const float *p_scale_y,
    float *p_output_y, int M, int N, int K, int lda, int ldb, int ldd) {

  EXT_ENFORCE(has_scales || p_scale_a == nullptr || *p_scale_a == 1,
              "scale_A must be empty or one for float");
  EXT_ENFORCE(has_scales || p_scale_b == nullptr || *p_scale_b == 1,
              "scale_B must be empty or one for float.");
  EXT_ENFORCE(has_scales_Y || p_scale_y == nullptr || *p_scale_y == 1,
              "scale_Y must be empty or one for float.");

  /*
  std::cout << "ComputeGemm("
            << "transa=" << transa << " transb=" << transb
            << " rowMajor_=" << rowMajor_ << " lda=" << lda << " ldb=" << ldb
            << " ldd=" << ldd << " M=" << M << " N=" << N << " K=" << K
            << ")\n";
  */
#if defined(__MACOSX__) || defined(__APPLE__)

  int i, j, k;
  int MN = M * N;
  if (p_input_c == nullptr) {
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = 0;
    }
  } else {
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = beta_ * p_input_c[i];
    }
  }

  if (rowMajor_ == 1) {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
            }
          }
        }
      } else {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
          }
        }
      }
    }
  } else {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[i * lda + k];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      } else {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[k * lda + i];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    }
  }

#else

  int i, j, k;
  int MN = M * N;
  if (p_input_c == nullptr) {
#pragma omp parallel for
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = 0;
    }
  } else {
#pragma omp parallel for
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = beta_ * p_input_c[i];
    }
  }

  if (rowMajor_ == 1) {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
#pragma omp parallel for
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
            }
          }
        }
      } else {
#pragma omp parallel for
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
#pragma omp parallel for
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
#pragma omp parallel for
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
          }
        }
      }
    }
  } else {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
#pragma omp parallel for
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[i * lda + k];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      } else {
#pragma omp parallel for
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
#pragma omp parallel for
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
#pragma omp parallel for
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[k * lda + i];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    }
  }

#endif
}

} // namespace ortops
