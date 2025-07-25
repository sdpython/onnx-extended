#include <gtest/gtest.h>
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

TEST(ortcy, inference) {
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  EXPECT_NE(api, nullptr);
  Ort::Env env;
  auto ort_env = &env; // std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);

  // requires C++ 17
  std_string_type model = get_data_path("ut_ortcy/data/add.onnx");

  Ort::Session session(*ort_env, model.c_str(), session_options);

  const char *input_names[] = {"X", "Y"};
  const char *output_names[] = {"Z"};

  float vector_1_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  int64_t vector_1_dim[] = {6, 1};

  float vector_2_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 50.f};
  int64_t vector_2_dim[] = {6, 1};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 6, vector_1_dim, 2),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 2)};

  Ort::RunOptions run_options;
  auto output_tensors =
      session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  const auto &vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  const float *floats_output = static_cast<const float *>(vector_filterred.GetTensorRawData());
  EXPECT_EQ(floats_output[0], 0);
  EXPECT_EQ(floats_output[1], 2);
  EXPECT_EQ(floats_output[2], 4);
  EXPECT_EQ(floats_output[3], 6);
  EXPECT_EQ(floats_output[4], 8);
  EXPECT_EQ(floats_output[5], 55);
}
