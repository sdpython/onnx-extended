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

TEST(ThreadPoolTests, CreateAndDestroy) {
  ThreadPool pool;
  pool.Start(4);
  EXPECT_EQ(pool.GetThreadCount(), 4);
}

TEST(ThreadPoolTests, SubmitSingleTask) {
  ThreadPool pool;
  pool.Start(2);
  int result = 0;
  auto task = [&result]() { result = 42; };
  pool.SubmitTask(task);
  pool.Wait();
  EXPECT_EQ(result, 42);
}

TEST(ThreadPoolTests, SubmitMultipleTasks) {
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

TEST(ThreadPoolTests, ParallelExecution) {
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
