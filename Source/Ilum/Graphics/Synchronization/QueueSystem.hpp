#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;
class Queue;

enum class QueueUsage
{
	Graphics,
	Compute,
	Transfer,
	Present
};

class QueueSystem
{
  public:
	QueueSystem();

	~QueueSystem() = default;

	void waitAll() const;

	Queue *acquire(QueueUsage usage = QueueUsage::Graphics);

	const std::vector<scope<Queue>> &getQueues() const;

  private:
	std::vector<scope<Queue>> m_queues;
	std::vector<Queue *>      m_graphics_queues;
	std::vector<Queue *>      m_present_queues;
	std::vector<Queue *>      m_compute_queues;
	std::vector<Queue *>      m_transfer_queues;
};
}        // namespace Ilum