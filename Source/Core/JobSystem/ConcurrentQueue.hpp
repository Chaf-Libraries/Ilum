#pragma once

#include "SpinLock.hpp"

#include <atomic>
#include <mutex>

namespace Ilum::Core
{
// Concurrent queue = Thread safe ring buffer
template <typename T, size_t capacity>
class ConcurrentQueue
{
  public:
	ConcurrentQueue() = default;

	~ConcurrentQueue() = default;

	ConcurrentQueue(const ConcurrentQueue &) = delete;

	ConcurrentQueue &operator=(const ConcurrentQueue &) = delete;

	void Push(const T &data)
	{
		size_t current_tail             = m_run_tail.fetch_add(1, std::memory_order_relaxed);
		m_data[current_tail % capacity] = data;
		m_tail.fetch_add(1, std::memory_order_relaxed);
	}

	bool TryPop(T &data)
	{
		size_t current_head = 0;

		{
			m_lock.Lock();
			if (m_tail <= m_head)
			{
				return false;
			}
			current_head = m_head++;
			m_lock.Unlock();
		}

		data = m_data[current_head % capacity];
		return true;
	}

	bool Empty() const
	{
		return m_tail <= m_head;
	}

  private:
	T                   m_data[capacity] = {};
	size_t              m_head           = 0;
	std::atomic<size_t> m_tail           = 0;
	std::atomic<size_t> m_run_tail       = 0;
	SpinLock            m_lock;
};
}        // namespace Ilum::Core