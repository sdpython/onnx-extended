#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
// #include "onnx_extended/ortcy/wrap/ortapi.h"
#include "onnxruntime_cxx_api.h"
#if __cplusplus >= 201703L
#include <filesystem>
#endif
#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

#define MODEL_TREE "ut_ortops/data/plot_op_tree_ensemble_sparse-f500-10-d10-s0.99.onnx"

void test_inference_tree_ensemble() {
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_THROW(api != nullptr);
  Ort::Env env;
  auto ort_env = &env; // std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  // session_options.Add(v2_domain);
  session_options.SetLogSeverityLevel(0);

// requires C++ 17
#if ((!defined(PYTHON_MANYLINUX) || !PYTHON_MANYLINUX) && __cplusplus >= 201703L)
  std::filesystem::path cwd = TEST_FOLDER;
#ifdef _WIN32
  std::wstring model = (cwd / MODEL_TREE).wstring();
#else
  std::string model = (cwd / MODEL_TREE).string();
#endif
#else
  std::string cwd = TEST_FOLDER;
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
  std::wstring model(cvt.from_bytes(cwd + std::string("/") + std::string(MODEL_TREE)));
#else
  std::string model = cwd + std::string("/") + std::string(MODEL_TREE);
#endif
#endif

  Ort::Session session(*ort_env, model.c_str(), session_options);

  const char *input_names[] = {"X"};
  const char *output_names[] = {"variable"};

  int64_t vector_1_dim[] = {100, 500};
  std::vector<float> vector_1_value(vector_1_dim[0] * vector_1_dim[1]);
  for (size_t i = 0; i < vector_1_value.size(); ++i) {
    vector_1_value[i] = 1.0f / static_cast<float>(i + 1);
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {Ort::Value::CreateTensor<float>(
      memory_info, vector_1_value.data(), vector_1_value.size(), vector_1_dim, 2)};

  Ort::RunOptions run_options;
  auto output_tensors =
      session.Run(run_options, input_names, input_tensors, 1, output_names, 1);
  const auto &vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  ASSERT_EQUAL(type_shape_info.GetDimensionsCount(), 2);
  const float *floats_output = static_cast<const float *>(vector_filterred.GetTensorRawData());
  ASSERT_EQUAL(floats_output[0], 0);
}

int main(int, char **) { test_inference_tree_ensemble(); }
