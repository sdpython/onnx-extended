#pragma once

#include "onnx_extended_helpers.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace onnx2 {
namespace utils {

class ThreadPool {
public:
  ThreadPool();
  ~ThreadPool();
  void Start(size_t num_threads);
  void SubmitTask(std::function<void()> job);
  void Wait();
  inline size_t GetThreadCount() const { return workers.size(); }
  inline bool IsStarted() const { return !workers.empty(); }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> jobs;

  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;

  void worker_thread();
};

} // namespace utils
} // namespace onnx2
