#include "JobSystem.hpp"

#include <queue>
#include <unordered_set>
#include <cassert>

namespace Ilum
{
SpinLock::SpinLock() :
    m_flag{ATOMIC_FLAG_INIT}
{
}

void SpinLock::Lock()
{
	while (m_flag.test_and_set(std::memory_order_acquire))
		;
}

void SpinLock::Unlock()
{
	m_flag.clear(std::memory_order_release);
}

ThreadPool::ThreadPool(uint32_t max_threads_num)
{
	for (uint32_t i = 0; i < max_threads_num; i++)
	{
		m_workers.emplace_back(
		    [this]() {
			    while (true)
			    {
				    std::function<void()> task;

				    {
					    std::unique_lock<std::mutex> lock(m_mutex);
					    m_condition.wait(lock, [this]() { return m_stop || !m_task_queue.Empty(); });
					    if (m_stop && m_task_queue.Empty())
					    {
						    return;
					    }
					    m_task_queue.TryPop(task);
				    }

				    task();
			    }
		    });
	}
}

ThreadPool ::~ThreadPool()
{
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_stop = true;
	}

	m_condition.notify_all();
	for (auto &worker : m_workers)
	{
		worker.join();
	}
}

size_t ThreadPool::GetThreadCount() const
{
	return m_workers.size();
}

void ThreadPool::WaitAll()
{
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_condition.notify_all();
	}

	for (auto &worker : m_workers)
	{
		if (worker.joinable())
		{
			worker.join();
		}
	}
}

JobNode::JobNode(std::function<void()> &&task) :
    m_task(task)
{
}

std::type_index JobNode::GetType()
{
	return typeid(JobNode);
}

void JobNode::Percede(JobNode *node)
{
	if (!node)
	{
		return;
	}

	m_successors.push_back(node);
	node->m_dependents.push_back(this);
	node->m_unfinish_dependents++;
}

void JobNode::Succeed(JobNode *node)
{
	if (!node)
	{
		return;
	}

	node->m_successors.push_back(this);
	this->m_dependents.push_back(node);
	this->m_unfinish_dependents++;
}

bool JobNode::Compile()
{
	return true;
}

void JobNode::Run()
{
	assert(m_unfinish_dependents == 0 && "Some dependents haven't completed yet");

	m_task();

	m_unfinish_dependents = static_cast<uint32_t>(m_dependents.size());

	for (auto *node : m_successors)
	{
		node->m_unfinish_dependents.fetch_sub(1, std::memory_order_relaxed);
	}
}

std::type_index JobGraph::GetType()
{
	return typeid(JobGraph);
}

JobGraph &JobGraph::AddNode(JobNode *node)
{
	m_nodes.push_back(node);
	return *this;
}

bool JobGraph::Compile()
{
	std::queue<JobNode *>  queue;
	std::vector<JobNode *> result;

	std::unordered_map<JobNode *, uint32_t> in_degree;

	for (auto *node : m_nodes)
	{
		if (in_degree.find(node) != in_degree.end())
		{
			return false;
		}

		in_degree[node] = static_cast<uint32_t>(node->m_dependents.size());
		if (in_degree[node] == 0)
		{
			queue.push(node);
		}
	}

	while (!queue.empty())
	{
		auto *t = queue.front();
		queue.pop();

		if (!t->Compile())
		{
			return false;
		}

		result.push_back(t);

		for (auto *node : t->m_successors)
		{
			in_degree[node]--;
			if (in_degree[node] == 0)
			{
				queue.push(node);
			}
		}
	}

	if (result.size() == m_nodes.size())
	{
		m_nodes = std::move(result);
		return true;
	}

	return false;
}

void JobGraph::Run()
{
	Compile();

	for (auto &node : m_nodes)
	{
		node->Run();
	}
}

JobSystem::JobSystem()
{
	m_thread_pool = std::make_unique<ThreadPool>(std::thread::hardware_concurrency() - 1);
}

JobSystem::~JobSystem()
{
	m_thread_pool.reset();
}

JobSystem &JobSystem::GetInstance()
{
	static JobSystem job_system;
	return job_system;
}

size_t JobSystem::GetThreadCount()
{
	return m_thread_pool->GetThreadCount();
}

void JobSystem::Execute(JobHandle &handle, JobGraph &graph)
{
	std::unordered_set<JobNode *> finished_nodes;

	handle.m_counter.fetch_add(static_cast<uint32_t>(graph.m_nodes.size()));

	while (handle.m_counter > 0)
	{
		for (auto &node : graph.m_nodes)
		{
			if (node->m_unfinish_dependents == 0 && finished_nodes.find(node) == finished_nodes.end())
			{
				finished_nodes.insert(node);

				if (node->GetType() == typeid(JobNode))
				{
					m_thread_pool->AddTask([&node, &handle]() {
						node->Run();
						handle.m_counter.fetch_sub(1);
					});
				}
				else
				{
					m_thread_pool->AddTask([&node, &handle, this]() {
						handle.m_counter.fetch_sub(1);
						Execute(handle, *static_cast<JobGraph *>(node));
					});
				}
			}
		}
	}
}

void JobSystem::Execute(JobHandle &handle, JobNode &node)
{
	handle.m_counter.fetch_add(1);

	m_thread_pool->AddTask([&node, &handle]() { node.Run();handle.m_counter.fetch_sub(1); });
}

void JobSystem::Dispatch(JobHandle &handle, uint32_t job_count, uint32_t group_size, const std::function<void(uint32_t)> &task)
{
	uint32_t group_count = (job_count + group_size - 1) / group_size;

	handle.m_counter.fetch_add(group_count);

	for (uint32_t group_id = 0; group_id < group_count; group_id++)
	{
		m_thread_pool->AddTask([task, &handle, group_id]() {
			task(group_id);
			handle.m_counter.fetch_sub(1);
		});
	}
}

bool JobSystem::IsBusy(const JobHandle &handle)
{
	return handle.m_counter.load() > 0;
}

void JobSystem::Wait(const JobHandle &handle)
{
	while (IsBusy(handle))
	{
		// wait...
	}
}

void JobSystem::WaitAll()
{
	m_thread_pool->WaitAll();
}
}