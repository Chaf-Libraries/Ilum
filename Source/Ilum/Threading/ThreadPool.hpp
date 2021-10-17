#pragma once

#include "Utils/PCH.hpp"
#include "Engine/Subsystem.hpp"

namespace Ilum
{
template <typename T>
class TQueue
{
  public:
	bool push(const T &value);

	bool pop(T &value);

	bool empty();

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

	const std::thread &getThread(size_t index) const;

	const std::vector<std::thread> &getThreads() const;

	// Clear queue
	void clear();

	std::function<void(size_t)> pop();

	template <typename F, typename... Args>
	auto addTask(F &&f, Args &&...args) -> std::future<decltype(f(0, args...))>;

	template <typename F>
	auto addTask(F &&f) -> std::future<decltype(f(0))>;

  private:
	size_t m_thread_count           = 0;
	size_t m_supported_thread_count = 0;

	std::vector<std::thread>        m_threads;
	std::vector<ref<std::atomic<bool>>>  m_flags;
	TQueue<std::function<void(size_t id)> *>         m_queue;
	std::unordered_map<std::thread::id, std::string> m_thread_names;
	std::atomic<size_t>                              m_waiting_count = 0;
	std::atomic<bool>                                m_stop          = false;
	std::atomic<bool>                                m_done          = false;
	std::mutex                                       m_mutex;
	std::condition_variable                          m_condition;
};
}        // namespace Ilum

#include "ThreadPool.inl"