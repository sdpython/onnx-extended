#include <gtest/gtest.h>
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
// #include "onnx_extended/ortcy/wrap/ortapi.h"
#include "onnxruntime_cxx_api.h"

TEST(ortops, inference_tree_ensemble) {
#if !defined(_WIN32) && (ORT_API_VERSION >= 17)
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_THROW(api != nullptr);
  Ort::Env env;
  auto ort_env = &env;
  Ort::SessionOptions session_options;
  session_options.RegisterCustomOpsLibrary(to_std_string_path(TESTED_CUSTOM_OPS_DLL).c_str());

  // requires C++ 17
  std_string_type model =
      get_data_path("ut_ortops/data/plot_op_tree_ensemble_implementations_custom.onnx");

  Ort::Session session(*ort_env, model.c_str(), session_options);
  // It needs to revisited.
  return;

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

  const char *env_p = std::getenv("LONG");
  bool long_test = env_p != nullptr && env_p[0] == '1';

  Ort::RunOptions run_options;
  for (int i = 0; i < (long_test ? 100000 : 1); ++i) {
    if (i > 0 && i % 10000 == 0)
      printf("i=%d\n", i);
    auto out = session.Run(run_options, input_names, input_tensors, 1, output_names, 1);
    EXPECT_EQ(out.size(), 1);
  }
  auto output_tensors =
      session.Run(run_options, input_names, input_tensors, 1, output_names, 1);
  const auto &vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  EXPECT_EQ(type_shape_info.GetDimensionsCount(), 2);
  const float *floats_output = static_cast<const float *>(vector_filterred.GetTensorRawData());
  // EXPECT_EQ(floats_output[0], 0);
  EXPECT_NE(floats_output, nullptr);
#endif
}
