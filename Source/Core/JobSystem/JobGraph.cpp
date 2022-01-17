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

std::type_index JobNode::type() const
{
	return typeid(JobNode);
}

void JobNode::percede(JobNode *node)
{
	if (!node)
	{
		return;
	}

	m_successors.push_back(node);
	node->m_dependents.push_back(this);
	node->m_unfinish_dependents++;
}

void JobNode::succeed(JobNode *node)
{
	if (!node)
	{
		return;
	}

	node->m_successors.push_back(this);
	this->m_dependents.push_back(node);
	this->m_unfinish_dependents++;
}

bool JobNode::compile()
{
	return true;
}

void JobNode::run()
{
	assert(m_unfinish_dependents == 0 && "Some dependents haven't completed yet");

	m_task();

	m_unfinish_dependents = static_cast<uint32_t>(m_dependents.size());

	for (auto *node : m_successors)
	{
		node->m_unfinish_dependents.fetch_sub(1, std::memory_order_relaxed);
	}
}

std::type_index JobGraph::type() const
{
	return typeid(JobGraph);
}

JobGraph &JobGraph::addNode(JobNode *node)
{
	m_nodes.push_back(node);
	return *this;
}

bool JobGraph::compile()
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

		if (!t->compile())
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

void JobGraph::run()
{
	compile();
	
	for (auto& node : m_nodes)
	{
		node->run();
	}
}
}        // namespace Ilum::Core