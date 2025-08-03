#include "thread_pool.h"

namespace onnx2 {
namespace utils {

ThreadPool::ThreadPool() {}

void ThreadPool::Start(size_t num_threads) {
  EXT_ENFORCE(workers.size() == 0, "ThreadPool already started");
  EXT_ENFORCE(num_threads > 0, "Number of threads must be greater than zero");
  stop = false;

  for (size_t i = 0; i < num_threads; ++i) {
    workers.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::SubmitTask(std::function<void()> job) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    jobs.push(std::move(job));
  }
  condition.notify_one();
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> job;

    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition.wait(lock, [this]() { return stop || !jobs.empty(); });

      if (stop)
        return;
      if (jobs.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      job = std::move(jobs.front());
      jobs.pop();
    }

    job();
  }
}

void ThreadPool::Wait() {
  while (jobs.size() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  stop = true;
  condition.notify_all();
  for (std::thread &worker : workers) {
    if (worker.joinable())
      worker.join();
  }
}

ThreadPool::~ThreadPool() { Wait(); }

} // namespace utils
} // namespace onnx2
