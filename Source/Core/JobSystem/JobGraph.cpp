#include "JobGraph.hpp"

#include <cassert>
#include <queue>
#include <stdexcept>

namespace Ilum::Core
{
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

JobGraph &JobGraph::addNode(JobNode *node)
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
	
	for (auto& node : m_nodes)
	{
		node->Run();
	}
}
}        // namespace Ilum::Core