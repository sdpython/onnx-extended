#include "dynamic_quantize_linear.h"
#include "cpu/cast_fp8.h"
#include <Eigen/Dense>

using namespace Eigen;

namespace ortops {

//////////////////////
// Operator definition
//////////////////////

void *DynamicQuantizeLinearOp::CreateKernel(
    const OrtApi &api, const OrtKernelInfo *info) const noexcept {
  return std::make_unique<DynamicQuantizeLinearKernel>(api, info).release();
};

const char *DynamicQuantizeLinearOp::GetName() const noexcept {
  return "DynamicQuantizeLinear";
}

const char *DynamicQuantizeLinearOp::GetExecutionProviderType() const noexcept {
  return "CPUExecutionProvider";
}

size_t DynamicQuantizeLinearOp::GetInputTypeCount() const noexcept {
  return 1;
};

ONNXTensorElementDataType
DynamicQuantizeLinearOp::GetInputType(std::size_t /* index */) const noexcept {
  return input_type_;
}

size_t DynamicQuantizeLinearOp::GetOutputTypeCount() const noexcept {
  return 3;
};

ONNXTensorElementDataType
DynamicQuantizeLinearOp::GetOutputType(std::size_t index) const {
  switch (index) {
  case 0:
    return quant_type_;
  case 1:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case 2:
    return quant_type_;
  default:
    EXT_THROW("Unexpected output index=", (uint64_t)index, ".");
  }
}

////////////////////////
// Kernel implementation
////////////////////////

DynamicQuantizeLinearKernel::DynamicQuantizeLinearKernel(
    const OrtApi &api, const OrtKernelInfo *info) {
  ThrowOnError(api, api.KernelInfoGetAttribute_int64(info, "to", &to_));
}

void DynamicQuantizeLinearKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);
  Ort::ConstValue input_X = ctx.GetInput(0);
  Ort::TensorTypeAndShapeInfo type_shape_info =
      input_X.GetTensorTypeAndShapeInfo();
  int64_t elem_type = type_shape_info.GetElementType();
  int64_t count = type_shape_info.GetElementCount();

  std::vector<int64_t> dimensions = type_shape_info.GetShape();
  Ort::UnownedValue output = ctx.GetOutput(0, dimensions);
  uint8_t *out = output.GetTensorMutableData<uint8_t>();

  std::vector<int64_t> empty;
  Ort::UnownedValue scale = ctx.GetOutput(1, empty);
  float *ptr_scale = scale.GetTensorMutableData<float>();

  Ort::UnownedValue zero_point = ctx.GetOutput(2, empty);
  uint8_t *ptr_zero_point = zero_point.GetTensorMutableData<uint8_t>();

  switch (elem_type) {
  // case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  //   const uint16_t *ptr = input_X.GetTensorData<uint16_t>();
  //   Compute(count, ptr, out, *ptr_scale, *ptr_zero_point);
  //   break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    const float *ptr = input_X.GetTensorData<float>();
    ComputeInternal<float>(count, ptr, out, *ptr_scale, *ptr_zero_point);
  } break;
  default:
    EXT_THROW("Unsupported input type ", elem_type, ".");
  }
}

template <typename T>
void DynamicQuantizeLinearKernel_ComputeInternal(int64_t n_elements,
                                                 const T *input,
                                                 uint8_t *output, float &scale,
                                                 uint8_t &zero_point,
                                                 int64_t to) {
  EXT_THROW("ComputeInternal must be specialized for type ", typeid(T).name(),
            ".");
}

template <>
void DynamicQuantizeLinearKernel_ComputeInternal<float>(
    int64_t n_elements, const float *input, uint8_t *output, float &scale,
    uint8_t &zero_point, int64_t to) {

  switch (to) {
  case 17 /* ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN */: {
    typedef Map<const Matrix<float, Dynamic, 1>> tensor_type;
    tensor_type data(input, n_elements);
    float x2 = data.array().square().sum() / static_cast<float>(n_elements);
    // see
    // https://github.com/onnx/onnx/pull/5472/files#diff-58654fc95848ff55a66c1914dab72cf40ff7c22e92ef1ae5d85908b7c82a34a6R224
    float std8 = 100.057724f;
    zero_point = 0;
    scale = std::sqrt(x2) / std8;

    if (zero_point != 0) {
      EXT_THROW("zero_point must be null not ", zero_point,
                " for type FLOAT8E4M3FN.");
    }
    float_to_e4m3fn(n_elements, input, output, scale);
  } break;
  default:
    EXT_THROW("Unsupported output type to=", to, ".");
  }
}

template <typename T>
void DynamicQuantizeLinearKernel::ComputeInternal(int64_t n_elements,
                                                  const T *input,
                                                  uint8_t *output, float &scale,
                                                  uint8_t &zero_point) {
  DynamicQuantizeLinearKernel_ComputeInternal<T>(n_elements, input, output,
                                                 scale, zero_point, to_);
}

} // namespace ortops
