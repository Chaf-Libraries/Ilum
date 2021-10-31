#include "QueueSystem.hpp"
#include "Queue.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
QueueSystem::QueueSystem()
{
	std::unordered_set<VkQueue> all_queues;

	for (uint32_t i = 0; i < GraphicsContext::instance()->getLogicalDevice().getPresentQueues().size(); i++)
	{
		all_queues.insert(GraphicsContext::instance()->getLogicalDevice().getPresentQueues()[i]);
	}

	for (uint32_t i = 0; i < GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues().size(); i++)
	{
		all_queues.insert(GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues()[i]);
	}

	for (uint32_t i = 0; i < GraphicsContext::instance()->getLogicalDevice().getComputeQueues().size(); i++)
	{
		all_queues.insert(GraphicsContext::instance()->getLogicalDevice().getComputeQueues()[i]);
	}

	for (uint32_t i = 0; i < GraphicsContext::instance()->getLogicalDevice().getTransferQueues().size(); i++)
	{
		all_queues.insert(GraphicsContext::instance()->getLogicalDevice().getTransferQueues()[i]);
	}

	std::unordered_map<VkQueue, Queue *> lut;
	for (auto &queue : all_queues)
	{
		m_queues.push_back(createScope<Queue>(queue));
		lut[queue] = m_queues.back().get();
	}

	for (auto &queue : GraphicsContext::instance()->getLogicalDevice().getPresentQueues())
	{
		m_present_queues.push_back(lut[queue]);
	}
	for (auto &queue : GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues())
	{
		m_graphics_queues.push_back(lut[queue]);
	}
	for (auto &queue : GraphicsContext::instance()->getLogicalDevice().getTransferQueues())
	{
		m_transfer_queues.push_back(lut[queue]);
	}
	for (auto &queue : GraphicsContext::instance()->getLogicalDevice().getComputeQueues())
	{
		m_compute_queues.push_back(lut[queue]);
	}
}

void QueueSystem::waitAll() const
{
	for (auto &queue : m_queues)
	{
		queue->waitIdle();
	}
}

Queue *QueueSystem::acquire(QueueUsage usage)
{
	size_t index = 0;

	// All you can use are present queues
	if (usage == QueueUsage::Present)
	{
		if (m_present_queues.empty())
		{
			return nullptr;
		}

		auto *queue = m_present_queues[index];
		while (queue->isBusy())
		{
			index = (index + 1ull) % m_present_queues.size();
			queue = m_present_queues[index];
		}
		return queue;
	}

	// You can use both transfer queues and graphics queues
	if (usage == QueueUsage::Transfer)
	{
		auto *queue = !m_transfer_queues.empty() ? m_transfer_queues[index] : m_graphics_queues[index];
		while (queue->isBusy())
		{
			index = (index + 1ull) % m_transfer_queues.size();
			queue = m_transfer_queues[index];
		}
		return queue;
	}

	// You can use both compute queues and graphics queues
	if (usage == QueueUsage::Compute)
	{
		auto *queue = !m_compute_queues.empty() ? m_compute_queues[index] : m_graphics_queues[index];
		while (queue->isBusy())
		{
			index = (index + 1ull) % m_compute_queues.size();
			queue = m_compute_queues[index];
		}
		return queue;
	}

	return nullptr;
}

const std::vector<scope<Queue>> &QueueSystem::getQueues() const
{
	return m_queues;
}
}        // namespace Ilum