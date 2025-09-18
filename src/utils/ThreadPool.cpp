#include "utils/ThreadPool.h"
#include "utils/ConfigParser.h"

ThreadPool &getThreadPool() {
  static ThreadPool pool([]() {
    return ConfigParser::instance().value<int>("numThreads");
  }());
  return pool;
}

ThreadPool::ThreadPool(size_t numThreads) : stop_(false) {
  if (numThreads == 0) {
    numThreads = 1;
    spdlog::error("Warning: ThreadPool created with 0 threads, defaulting to 1.");
  }
  for (size_t i = 0; i < numThreads; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(
              lock, [this] { return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ && this->tasks_.empty())
            return;
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }

        task();

        {
          std::unique_lock<std::mutex> lock(this->completion_mutex_);
          this->completed_tasks_++;
          this->completion_condition_.notify_one();
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  condition_.notify_all();

  for (std::thread &worker : workers_) {
    worker.join();
  }
}

void ThreadPool::run_batch(std::vector<std::function<void()>> tasks) {
  size_t batchSize = tasks.size();
  if (batchSize == 0)
    return;

  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_)
      throw std::runtime_error("run_batch called on stopped ThreadPool");

    for (auto &task : tasks) {
      tasks_.push(std::move(task));
    }
    expected_tasks_ += batchSize;
  }

  condition_.notify_all();

  {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_condition_.wait(lock, [this, expected = expected_tasks_.load()] {
      return completed_tasks_.load() >= expected;
    });
  }
}
