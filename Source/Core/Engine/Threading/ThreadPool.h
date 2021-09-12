#pragma once

#include "Core/Engine/PCH.hpp"
#include "Core/Engine/Subsystem.hpp"

namespace Ilum
{
template <typename T>
class TQueue
{
  public:
	bool push(const T &value)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		this->m_queue.push(value);
		return true;
	}

	bool pop(T &value)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		if (m_queue.empty())
		{
			return false;
		}
		value = m_queue.front();
		m_queue.pop();
		return true;
	}

	bool empty()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		return m_queue.empty();
	}

  private:
	std::queue<T> m_queue;
	std::mutex    m_mutex;
};

class ThreadPool : public TSubsystem<ThreadPool>
{
  public:
	ThreadPool(Context *context = nullptr);

	~ThreadPool();

	size_t size() const;

	size_t idleCount() const;

	std::thread &getThread(size_t index);

	// Clear queue
	void clear();

	std::function<void(size_t)> pop();

	template <typename F, typename... Args>
	auto addTask(F &&f, Args &&...args) -> std::future<decltype(f(0, args...))>
	{
		auto pack = std::make_shared<std::packaged_task<decltype(f(0, args...))(size_t)>>(std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...));
		auto func = new std::function<void(size_t id)>([pack](size_t id) { (*pack)(id); });
		m_queue.push(func);
		std::unique_lock<std::mutex> lock(m_mutex);
		m_condition.notify_one();
		return pack->get_future();
	}

	template <typename F>
	auto addTask(F &&f) -> std::future<decltype(f(0))>
	{
		auto pack = std::make_shared<std::packaged_task<decltype(f(0))(size_t)>>(std::forward<F>(f));
		auto func = new std::function<void(size_t id)>([pack](size_t id) { (*pack)(id); });
		m_queue.push(func);
		std::unique_lock<std::mutex> lock(m_mutex);
		m_condition.notify_one();
		return pack->get_future();
	}

  private:
	size_t m_thread_count           = 0;
	size_t m_supported_thread_count = 0;

	std::vector<std::unique_ptr<std::thread>>        m_threads;
	std::vector<std::shared_ptr<std::atomic<bool>>>  m_flags;
	TQueue<std::function<void(size_t id)> *>         m_queue;
	std::unordered_map<std::thread::id, std::string> m_thread_names;
	std::atomic<size_t>                              m_waiting_count = 0;
	std::atomic<bool>                                m_stop          = false;
	std::atomic<bool>                                m_done          = false;
	std::mutex                                       m_mutex;
	std::condition_variable                          m_condition;
};
}        // namespace Ilum