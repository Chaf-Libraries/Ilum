#pragma once

#include "ConcurrentQueue.hpp"

#include <future>
#include <functional>
#include <unordered_map>

namespace Ilum::Core
{
class ThreadPool
{
  public:
	ThreadPool(uint32_t max_threads_num);

	~ThreadPool();

	size_t GetThreadCount() const;

	template <typename Task, typename... Args>
	inline auto AddTask(Task &&task, Args &&...args)
	    -> std::future<decltype(task(args...))>
	{
		using return_type = decltype(task(args...));

		auto pack = std::make_shared<std::packaged_task<return_type()>>(
		    std::bind(std::forward<Task>(task), std::forward<Args>(args)...));
		m_task_queue.Push([pack]() { (*pack)(); });

		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_condition.notify_one();
		}

		return pack->get_future();
	}

  private:
	ConcurrentQueue<std::function<void()>, 1024>       m_task_queue;
	std::unordered_map<std::thread::id, const char *> m_thread_names;

	std::vector<std::thread> m_workers;
	std::mutex               m_mutex;
	std::atomic<bool>        m_stop = false;
	std::condition_variable  m_condition;
};
}        // namespace Ilum::Core