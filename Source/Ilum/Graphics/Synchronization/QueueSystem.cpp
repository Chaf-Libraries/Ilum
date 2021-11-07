#include "QueueSystem.hpp"
#include "Queue.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

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
	uint32_t i = 0;
	for (auto &queue : GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues())
	{
		m_graphics_queues.push_back(lut[queue]);
		VK_Debugger::setName(*lut[queue], std::to_string(i++).c_str());
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

void QueueSystem::waitAll()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	vkDeviceWaitIdle(GraphicsContext::instance()->getLogicalDevice());
}

Queue *QueueSystem::acquire(QueueUsage usage)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (usage == QueueUsage::Present)
	{
		if (m_present_queues.empty())
		{
			return nullptr;
		}

		auto *queue = m_present_queues[m_present_index];
		//LOG_INFO("Present Index - {}", m_present_index);
		m_present_index = (m_present_index + 1ull) % m_present_queues.size();
		queue->waitIdle();
		return queue;
	}

	if (usage == QueueUsage::Graphics)
	{
		if (m_graphics_queues.empty())
		{
			return nullptr;
		}

		auto *queue     = m_graphics_queues[m_graphics_index];
		//LOG_INFO("Graphics Index - {}", m_graphics_index);
		m_graphics_index = (m_graphics_index + 1ull) % m_graphics_queues.size();
		queue->waitIdle();
		return queue;
	}

	if (usage == QueueUsage::Transfer)
	{
		if (m_transfer_queues.empty())
		{
			return nullptr;
		}

		auto *queue      = m_transfer_queues[m_transfer_index];
		//LOG_INFO("Transfer Index - {}", m_transfer_index);
		m_transfer_index = (m_transfer_index + 1ull) % m_transfer_queues.size();
		queue->waitIdle();
		return queue;
	}

	if (usage == QueueUsage::Compute)
	{
		if (m_compute_queues.empty())
		{
			return nullptr;
		}

		auto *queue      = m_compute_queues[m_compute_index];
		//LOG_INFO("Compute Index - {}", m_compute_index);
		m_compute_index = (m_compute_index + 1ull) % m_compute_queues.size();
		queue->waitIdle();
		return queue;
	}

	return nullptr;
}

const std::vector<scope<Queue>> &QueueSystem::getQueues() const
{
	return m_queues;
}
}        // namespace Ilum