#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class SpinLock
{
  public:
	SpinLock();
	~SpinLock() = default;

	void Lock();
	void Unlock();

  private:
	std::atomic_flag m_flag;
};

template <typename T, size_t capacity>
class RingBuffer
{
  public:
	RingBuffer()  = default;
	~RingBuffer() = default;

	RingBuffer(const RingBuffer &) = delete;
	RingBuffer &operator=(const RingBuffer &) = delete;
	RingBuffer(RingBuffer &&)                 = delete;
	RingBuffer &operator=(RingBuffer &&) = delete;

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

class ThreadPool
{
  public:
	ThreadPool(uint32_t max_threads_num);
	~ThreadPool();

	ThreadPool(const ThreadPool &) = delete;
	ThreadPool &operator=(const ThreadPool &) = delete;
	ThreadPool(ThreadPool &&)                 = delete;
	ThreadPool &operator=(ThreadPool &&) = delete;

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

	void WaitAll();

  private:
	RingBuffer<std::function<void()>, 1024>           m_task_queue;
	std::unordered_map<std::thread::id, const char *> m_thread_names;

	std::vector<std::thread> m_workers;
	std::mutex               m_mutex;
	std::atomic<bool>        m_stop = false;
	std::condition_variable  m_condition;
};

class JobNode
{
	friend class JobSystem;
	friend class JobGraph;

  public:
	// Job Graph won't provide return value
	explicit JobNode(std::function<void()> &&task);

	JobNode() = default;

	~JobNode() = default;

	virtual std::type_index GetType();

	// A.percede(B) => B depend on A
	void Percede(JobNode *node);

	// A.succeed(B) => A depend on B
	void Succeed(JobNode *node);

	virtual bool Compile();

	virtual void Run();

  protected:
	std::vector<JobNode *> m_successors;
	std::vector<JobNode *> m_dependents;
	std::atomic<uint32_t>  m_unfinish_dependents = 0;

  private:
	std::function<void()> m_task;
};

class JobGraph : public JobNode
{
	friend class JobSystem;

  public:
	explicit JobGraph() = default;

	~JobGraph() = default;

	virtual std::type_index GetType() override;

	JobGraph &AddNode(JobNode *node);

	// Validation and topology sorting
	virtual bool Compile() override;

	// Single thread job graph execute
	// You must compile job graph before it runs
	virtual void Run() override;

  private:
	std::vector<JobNode *> m_nodes;
};

class JobSystem;

class JobHandle
{
	friend class JobSystem;
	// The number of subjob current job is dealing
	std::atomic<uint32_t> m_counter = 0;
};

class EXPORT_API JobSystem
{
  public:
	JobSystem();

	~JobSystem();

	static JobSystem &GetInstance();

	size_t GetThreadCount();

	// You must compile job graph before it executed
	void Execute(JobHandle &handle, JobGraph &graph);
	void Execute(JobHandle &handle, JobNode &node);

	// Async execute, can be used in resource loading
	template <typename Task, typename... Args>
	inline auto Execute(JobHandle &handle, Task &&task, Args &&...args)
	    -> std::future<decltype(task(args...))>
	{
		handle.m_counter.fetch_add(1);

		return m_thread_pool->AddTask([task, &handle, args...]() {
			auto result = task(std::forward<Args>(args)...);
			handle.m_counter.fetch_sub(1);
			return result;
		});
	}

	// Using dispatch method, task need group id as parameter
	void Dispatch(JobHandle &handle, uint32_t job_count, uint32_t group_size, const std::function<void(uint32_t)> &task);
	bool IsBusy(const JobHandle &handle);
	void Wait(const JobHandle &handle);
	void WaitAll();

  private:
	std::unique_ptr<ThreadPool> m_thread_pool = nullptr;
};

}        // namespace Ilum