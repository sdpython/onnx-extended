#include "my_kernel_attr.h"

namespace ortops {

MyCustomKernelWithAttributes::MyCustomKernelWithAttributes(
    const OrtApi &api, const OrtKernelInfo *info) {
  // A float attribute.
  float value_float;
  ThrowOnError(
      api, api.KernelInfoGetAttribute_float(info, "att_float", &value_float));
  att_float = value_float;

  // An integer attribute.
  int64_t value_int64;
  ThrowOnError(
      api, api.KernelInfoGetAttribute_int64(info, "att_int64", &value_int64));
  att_int64 = value_int64;

  // A string attribute.
  char value_string[1000];
  std::size_t size = 1000;
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, "att_string",
                                                      value_string, &size));
  att_string = value_string;

  // A tensor attribute
  // Retrieve the value.
  OrtAllocator *cpu_allocator;
  ThrowOnError(api, api.GetAllocatorWithDefaultOptions(&cpu_allocator));

  OrtValue *value_tensor = nullptr;
  ThrowOnError(api, api.KernelInfoGetAttribute_tensor(
                        info, "att_tensor", cpu_allocator, &value_tensor));

  // Retrieve the dimensions and the element type.
  OrtTensorTypeAndShapeInfo *shape_info;
  ThrowOnError(api, api.GetTensorTypeAndShape(value_tensor, &shape_info));

  // Retrieve the element type.
  ONNXTensorElementDataType elem_type;
  ThrowOnError(api, api.GetTensorElementType(shape_info, &elem_type));
  if (elem_type !=
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    api.ReleaseTensorTypeAndShapeInfo(shape_info);
    api.ReleaseValue(value_tensor);
    throw std::runtime_error(
        "Attribute 'att_tensor' of operator 'MyCustomOpWithAttributes' expects "
        "a double tensor.");
  }

  // Retrieve the number of elements in the shape.
  std::size_t n_dims;
  ThrowOnError(api, api.GetDimensionsCount(shape_info, &n_dims));
  std::vector<int64_t> shape(n_dims);
  ThrowOnError(api, api.GetDimensions(shape_info, shape.data(), n_dims));

  std::size_t size_tensor;
  ThrowOnError(api, api.GetTensorShapeElementCount(shape_info, &size_tensor));
  att_tensor_double.resize(size_tensor);
  void *data;
  ThrowOnError(api, api.GetTensorMutableData(value_tensor, &data));

  memcpy(att_tensor_double.data(), data, size_tensor * sizeof(double));

  // Release the allocated objects.
  api.ReleaseTensorTypeAndShapeInfo(shape_info);
  api.ReleaseValue(value_tensor);

  // Verifications.
  if (att_tensor_double.empty()) {
    throw std::runtime_error("Attribute 'att_tensor' of operator "
                             "'MyCustomOpWithAttributes' cannot be empty.");
  }
}

void MyCustomKernelWithAttributes::Compute(OrtKernelContext *context) {
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

  const std::size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

  // Do computation
  double cst = att_tensor_double[0] + static_cast<double>(att_float) +
               static_cast<double>(att_int64) +
               static_cast<double>(att_string[0]);

  for (std::size_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i] + cst;
  }
}

void *MyCustomOpWithAttributes::CreateKernel(const OrtApi &api,
                                             const OrtKernelInfo *info) const {
  return std::make_unique<MyCustomKernelWithAttributes>(api, info).release();
}

const char *MyCustomOpWithAttributes::GetName() const {
  return "MyCustomOpWithAttributes";
}

const char *MyCustomOpWithAttributes::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

size_t MyCustomOpWithAttributes::GetInputTypeCount() const { return 2; };

ONNXTensorElementDataType
MyCustomOpWithAttributes::GetInputType(std::size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

size_t MyCustomOpWithAttributes::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType
MyCustomOpWithAttributes::GetOutputType(std::size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

} // namespace ortops
