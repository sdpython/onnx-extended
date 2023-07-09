#include "onnx_extended_test_common.h"
#include "onnx_extended/ortcy/wrap/ortapi.h"
#include "onnxruntime_cxx_api.h"
#include <filesystem>
#include <iostream>
#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

void testAssertTrue() {
  ASSERT_THROW( true );
}

void test_inference() {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_THROW(api != nullptr);
  Ort::Env env;
  auto ort_env = &env;// std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  // session_options.Add(v2_domain);
  session_options.SetLogSeverityLevel(0);

  // requires C++ 17
  #if __cplusplus >= 201703L 
    std::filesystem::path cwd = TEST_FOLDER;
    #ifdef _WIN32
    std::wstring model = (cwd / "ut_ortcy/data/add.onnx").wstring();
    #else
    std::string model = (cwd / "ut_ortcy/data/add.onnx").string();
    #endif
  #else
    std::string cwd = TEST_FOLDER;
    #ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
    std::wstring model(cvt.from_bytes(cwd + std::string("/") + std::string("ut_ortcy/data/add.onnx")));
    #else
    std::string model = cwd + std::string("/") + std::string("ut_ortcy/data/add.onnx");
    #endif
  #endif

  Ort::Session session(* ort_env, model.c_str(), session_options);

  const char* input_names[] = {"X", "Y"};
  const char* output_names[] = {"Z"};

  float vector_1_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  int64_t vector_1_dim[] = {6, 1};

  float vector_2_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 50.f};
  int64_t vector_2_dim[] = {6, 1};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 6, vector_1_dim, 2),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 2)};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  const auto& vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  const float* floats_output = static_cast<const float*>(vector_filterred.GetTensorRawData());
  ASSERT_EQUAL(floats_output[0], 0);
  ASSERT_EQUAL(floats_output[1], 2);
  ASSERT_EQUAL(floats_output[2], 4);
  ASSERT_EQUAL(floats_output[3], 6);
  ASSERT_EQUAL(floats_output[4], 8);
  ASSERT_EQUAL(floats_output[5], 55);
}


int main(int, char**) {
  testAssertTrue();
  test_inference();
}
