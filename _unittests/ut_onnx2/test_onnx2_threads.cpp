#include "onnx_extended/onnx2/cpu/onnx2.h"
#include "onnx_extended/onnx2/cpu/thread_pool.h"
#include "onnx_extended_helpers.h"
#include "onnx_extended_test_common.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace onnx2;
using namespace onnx2::utils;

TEST(onnx2_threads, CreateAndDestroy) {
  ThreadPool pool;
  pool.Start(4);
  EXPECT_EQ(pool.GetThreadCount(), 4);
}

TEST(onnx2_threads, SubmitSingleTask) {
  ThreadPool pool;
  pool.Start(2);
  int result = 0;
  auto task = [&result]() {
    for (size_t i = 0; i < 42; ++i) {
      result += 1;
    }
  };
  pool.SubmitTask(task);
  pool.Wait();
  EXPECT_EQ(result, 42);
}

TEST(onnx2_threads, SubmitMultipleTasks) {
  ThreadPool pool;
  pool.Start(4);
  constexpr int num_tasks = 100;
  std::atomic<int> counter(0);
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
  }
  pool.Wait();
  EXPECT_EQ(counter.load(), num_tasks);
}

TEST(onnx2_threads, ParallelExecution) {
  // Utiliser plus de threads que les cœurs physiques pour tester la répartition
  ThreadPool pool;
  pool.Start(8);

  std::atomic<int> counter(0);
  std::vector<int> thread_ids;
  std::mutex mutex;

  constexpr int num_tasks = 20;
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter, &thread_ids, &mutex]() {
      int thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
      {
        std::lock_guard<std::mutex> lock(mutex);
        thread_ids.push_back(thread_id);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      counter.fetch_add(1, std::memory_order_relaxed);
    });
  }

  pool.Wait();

  EXPECT_EQ(counter.load(), num_tasks);

  std::sort(thread_ids.begin(), thread_ids.end());
  auto unique_end = std::unique(thread_ids.begin(), thread_ids.end());
  int unique_threads = std::distance(thread_ids.begin(), unique_end);
  EXPECT_GT(unique_threads, 1);
}

TEST(onnx2_threads, ParallelModelProcessing0) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto &graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto &tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 0;
    ModelProto model_proto2;
    stream.StartThreadPool(0);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(),
              model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}

TEST(onnx2_threads, ParallelModelProcessing4) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto &graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto &tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    ModelProto model_proto2;
    stream.StartThreadPool(2);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(),
              model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}
