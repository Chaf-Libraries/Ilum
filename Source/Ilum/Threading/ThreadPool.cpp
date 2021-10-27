#include "ThreadPool.hpp"

namespace Ilum
{
ThreadPool::ThreadPool(Context *context) :
    TSubsystem<ThreadPool>(context)
{
	// Max thread number CPU supports
	m_supported_thread_count = std::thread::hardware_concurrency();

	// Leave one for main thread
	m_thread_count                             = m_supported_thread_count - 1;
	m_thread_names[std::this_thread::get_id()] = "main";

	m_threads.resize(m_thread_count);
	m_flags.resize(m_thread_count);

	for (uint32_t i = 0; i < m_thread_count; i++)
	{
		m_flags[i] = std::make_shared<std::atomic<bool>>(false);

		ref<std::atomic<bool>> flag(m_flags[i]);
		auto                               f = [this, i, flag]() {
            std::atomic<bool> &             _flag = *flag;
            std::function<void(size_t id)> *_f;
            bool                            is_pop = m_queue.pop(_f);
            while (true)
            {
                while (is_pop)
                {
                    scope<std::function<void(size_t id)>> func(_f);
                    (*_f)(i);
                    if (_flag)
                    {
                        return;
                    }
                    else
                    {
                        is_pop = m_queue.pop(_f);
                    }
                }
                std::unique_lock<std::mutex> lock(m_mutex);
                ++m_waiting_count;
                m_condition.wait(lock, [this, &_f, &is_pop, &_flag]() { is_pop = m_queue.pop(_f); return is_pop||_flag||m_done; });
                --m_waiting_count;
                if (!is_pop)
                {
                    return;
                }
            }
		};
		m_threads[i] = std::thread(f);
		m_thread_names[m_threads[i].get_id()] = "worker_" + std::to_string(i);

		LOG_INFO("Create thread [{}]: {}", i, m_thread_names[m_threads[i].get_id()]);
	}
	LOG_INFO("{} threads have been created", m_thread_count);
}

ThreadPool::~ThreadPool()
{
	m_done = true;

	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_condition.notify_all();
	}

	for (uint32_t i = 0; i < m_threads.size(); i++)
	{
		if (m_threads[i].joinable())
			m_threads[i].join();
	}

	clear();
	m_threads.clear();
	m_flags.clear();
}

size_t ThreadPool::size() const
{
	return m_threads.size();
}

size_t ThreadPool::idleCount() const
{
	return m_waiting_count;
}

std::thread &ThreadPool::getThread(size_t index)
{
	return m_threads[index];
}

const std::thread &ThreadPool::getThread(size_t index) const
{
	return m_threads[index];
}

const std::vector<std::thread> &ThreadPool::getThreads() const
{
	return m_threads;
}

uint32_t ThreadPool::threadIndex(const std::thread::id &thread_id)
{
	for (uint32_t i = 0; i < m_threads.size(); i++)
	{
		if (thread_id == m_threads[i].get_id())
		{
			return i;
		}
	}
	return std::numeric_limits<uint32_t>::max();
}

void ThreadPool::clear()
{
	std::function<void(size_t id)> *_f;
	while (m_queue.pop(_f))
	{
		delete _f;
	}
}

std::function<void(size_t)> ThreadPool::pop()
{
	std::function<void(size_t id)> *_f = nullptr;
	m_queue.pop(_f);
	scope<std::function<void(size_t id)>> func(_f);
	std::function<void(size_t)>                     f;
	if (_f)
	{
		f = *_f;
	}
	return f;
}
}        // namespace Ilum