#pragma once

#include "JobGraph.hpp"
#include "ThreadPool.hpp"

namespace Ilum::Core
{
class JobSystem;

class JobHandle
{
	friend class JobSystem;
	// The number of subjob current job is dealing
	std::atomic<uint32_t> m_counter = 0;
};

class JobSystem
{
  public:
	static void initialize();

	static size_t getThreadCount();

	// You must compile job graph before it executed
	static void execute(JobHandle &handle, JobGraph &graph);

	static void execute(JobHandle &handle, JobNode &node);

	// Async execute, can be used in resource loading
	template <typename Task, typename... Args>
	static auto execute(JobHandle &handle, Task &&task, Args &&...args)
	    -> std::future<decltype(task(args...))>
	{
		handle.m_counter.fetch_add(1);

		return instance().m_thread_pool->addTask([&task, &handle, &args...]() {
			auto result = task(std::forward<Args>(args)...);
			handle.m_counter.fetch_sub(1);
			return result;
		});
	}

	// Using dispatch method, task need group id as parameter
	static void dispatch(JobHandle &handle, uint32_t job_count, uint32_t group_size, const std::function<void(uint32_t)> &task);

	static bool isBusy(const JobHandle &handle);

	static void wait(const JobHandle &handle);

  private:
	JobSystem() = default;

	~JobSystem() = default;

	static JobSystem &instance();

  private:
	std::unique_ptr<ThreadPool> m_thread_pool = nullptr;
};
}        // namespace Ilum::Core