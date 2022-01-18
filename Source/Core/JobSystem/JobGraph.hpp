#pragma once

#include <atomic>
#include <functional>
#include <typeindex>
#include <vector>

namespace Ilum::Core
{
class JobSystem;
class JobGraph;

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

	JobGraph &addNode(JobNode *node);

	// Validation and topology sorting
	virtual bool Compile() override;

	// Single thread job graph execute
	// You must compile job graph before it runs
	virtual void Run() override;

  private:
	std::vector<JobNode *> m_nodes;
};
}        // namespace Ilum::Core