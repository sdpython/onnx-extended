#include "custom_tree_assembly.h"
#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__MACOSX__)
#define USE_DLFCN
#include <dlfcn.h>
#endif

namespace ortops {

template <typename T, int32_t Rank> struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};

typedef int32_t (*InitModelFn)();

class TreebeardSORunner {
  void *so;
  void *predFnPtr;
  int32_t batchSize;
  int32_t rowSize;

#ifdef USE_DLFCN
  void CallFuncAndGetIntValueFromSo(const std::string &functionName, int32_t &field) {
    using GetFunc_t = int32_t (*)();
    auto get = reinterpret_cast<GetFunc_t>(dlsym(so, functionName.c_str()));
    field = get();
  }
#else
  void CallFuncAndGetIntValueFromSo(const std::string &, int32_t &) {
    EXT_THROW("CallFuncAndGetIntValueFromSo: only works on linux.");
  }
#endif

public:
  TreebeardSORunner(const char *soFilePath) {
#ifdef USE_DLFCN
    so = dlopen(soFilePath, RTLD_NOW);
#else
    so = nullptr;
#endif
    if (!so) {
      EXT_THROW("Failed to load so '", soFilePath, "'.");
    }
#ifdef USE_DLFCN
    InitModelFn initModelFnPtr = (InitModelFn)dlsym(so, "Init_model");
#else
    InitModelFn initModelFnPtr = nullptr;
#endif
    if (!initModelFnPtr) {
      EXT_THROW("Failed to load 'Init_model' function from so '", soFilePath, ".");
    }
    initModelFnPtr();
    CallFuncAndGetIntValueFromSo("GetBatchSize", batchSize);
    CallFuncAndGetIntValueFromSo("GetRowSize", rowSize);

#ifdef USE_DLFCN
    predFnPtr = dlsym(so, "Prediction_Function");
#endif
    if (!predFnPtr) {
      EXT_THROW("Failed to load 'Prediction_Function' function from so '", soFilePath, ".");
    }
  }

  ~TreebeardSORunner() {
#ifdef USE_DLFCN
    dlclose(so);
#endif
  }

  int32_t GetBatchSize() const { return batchSize; }
  int32_t GetRowSize() const { return rowSize; }

  template <typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue) {
    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(
        InputElementType *, InputElementType *, int64_t, int64_t, int64_t, int64_t, int64_t,
        ReturnType *, ReturnType *, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predFnPtr);

    InputElementType *ptr = input;
    InputElementType *alignedPtr = input;

    ReturnType *resultPtr = returnValue;
    ReturnType *resultAlignedPtr = returnValue;

    int64_t offset = 0, stride = 1;
    int64_t resultLen = batchSize;

    inferenceFuncPtr(ptr, alignedPtr, offset, batchSize, rowSize, rowSize, stride, resultPtr,
                     resultAlignedPtr, offset, resultLen, stride);
    return 0;
  }
};

//////////////////
// CustomTreeAssemblyOp...
//////////////////

void *CustomTreeAssemblyOp::CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
  return std::make_unique<CustomTreeAssemblyKernel>(api, info, classifier_).release();
}

const char *CustomTreeAssemblyOp::GetName() const {
  return classifier_ ? "TreeEnsembleAssemblyClassifier" : "TreeEnsembleAssemblyRegressor";
}

const char *CustomTreeAssemblyOp::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

size_t CustomTreeAssemblyOp::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType CustomTreeAssemblyOp::GetInputType(std::size_t) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

OrtCustomOpInputOutputCharacteristic
CustomTreeAssemblyOp::GetInputCharacteristic(std::size_t) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

size_t CustomTreeAssemblyOp::GetOutputTypeCount() const { return classifier_ ? 2 : 1; }

ONNXTensorElementDataType CustomTreeAssemblyOp::GetOutputType(std::size_t index) const {
  if (classifier_) {
    switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      EXT_THROW("Output index=", (uint64_t)index, " is out of boundary.");
    }

  } else {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
}

OrtCustomOpInputOutputCharacteristic
CustomTreeAssemblyOp::GetOutputCharacteristic(std::size_t) const {
  return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

///////////////////
// CustomTreeAssemblyKernel
///////////////////

CustomTreeAssemblyKernel::CustomTreeAssemblyKernel(const OrtApi &api, const OrtKernelInfo *info,
                                                   bool classifier) {
  classifier_ = classifier;
  assembly_name_ = KernelInfoGetOptionalAttributeString(api, info, "assembly", "");
  EXT_ENFORCE(!assembly_name_.empty(), "Parameter 'assembly' cannot be empty.");
  TreebeardSORunner *tree_runner = new TreebeardSORunner(assembly_name_.c_str());
  assembly_runner_ = (void *)tree_runner;
}

CustomTreeAssemblyKernel::~CustomTreeAssemblyKernel() {
  TreebeardSORunner *tree_runner = (TreebeardSORunner *)assembly_runner_;
  // EXT_ENFORCE(tree_runner != nullptr);
  delete tree_runner;
}

void CustomTreeAssemblyKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  Ort::ConstValue X = ctx.GetInput(0);
  std::vector<int64_t> dimensions = X.GetTensorTypeAndShapeInfo().GetShape();
  TreebeardSORunner *tree_runner = (TreebeardSORunner *)assembly_runner_;
  EXT_ENFORCE(tree_runner != nullptr);
  EXT_ENFORCE(dimensions.size() == 2, "Input shape must have two dimensions.")
  EXT_ENFORCE(dimensions[0] == tree_runner->GetBatchSize() &&
                  dimensions[1] == (tree_runner->GetRowSize()),
              "The assembly was compiled for an input shape stricly equal to ",
              tree_runner->GetBatchSize(), "x", tree_runner->GetRowSize(),
              " but input shape is ", dimensions[0], "x", dimensions[1]);
  if (classifier_) {
    EXT_THROW("This kernel only supports regressors.")
  } else {
    Ort::UnownedValue output = ctx.GetOutput(0, {dimensions[0], 1});
    const float *inx = X.GetTensorData<float>();
    float *out = output.GetTensorMutableData<float>();
    tree_runner->RunInference(inx, out);
  }
}

} // namespace ortops
