#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

namespace ortops {

struct MyCustomKernelWithAttributes {
  MyCustomKernelWithAttributes(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

  private:
    std::string att_string;
    std::string att_float;
    int64_t att_int64;
    std::vector<double> att_tensor_double;
};

struct MyCustomOpWithAttributes : Ort::CustomOpBase<MyCustomOpWithAttributes, MyCustomKernelWithAttributes> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const ;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace ortops
