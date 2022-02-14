#include "QueueSystem.hpp"
#include "Queue.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

#include <Graphics/Vulkan.hpp>

namespace Ilum
{
QueueSystem::QueueSystem()
{
	std::unordered_set<VkQueue> all_queues;

	all_queues.insert(Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Present));
	all_queues.insert(Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Graphics));
	all_queues.insert(Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Compute));
	all_queues.insert(Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Transfer));

	std::unordered_map<VkQueue, Queue *> lut;
	for (auto &queue : all_queues)
	{
		m_queues.push_back(createScope<Queue>(queue));
		lut[queue] = m_queues.back().get();
	}

	m_present_queues.push_back(lut[Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Present)]);

	m_graphics_queues.push_back(lut[Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Graphics)]);

	m_transfer_queues.push_back(lut[Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Transfer)]);

	m_compute_queues.push_back(lut[Graphics::RenderContext::GetDevice().GetQueue(Graphics::QueueFamily::Compute)]);
}

void QueueSystem::waitAll()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	for (auto &queue : m_queues)
	{
		queue->waitIdle();
	}
}

Queue *QueueSystem::acquire(QueueUsage usage, uint32_t index)
{
	if (usage == QueueUsage::Present)
	{
		if (m_present_queues.empty())
		{
			return nullptr;
		}

		return m_present_queues[0];
	}

	if (usage == QueueUsage::Graphics)
	{
		if (m_graphics_queues.empty())
		{
			return nullptr;
		}

		return m_graphics_queues[0];
	}

	if (usage == QueueUsage::Transfer)
	{
		if (m_transfer_queues.empty())
		{
			return nullptr;
		}

		return m_transfer_queues[0];
	}

	if (usage == QueueUsage::Compute)
	{
		if (m_compute_queues.empty())
		{
			return nullptr;
		}

		return m_compute_queues[0];
	}

	return nullptr;
}

const std::vector<scope<Queue>> &QueueSystem::getQueues() const
{
	return m_queues;
}
}        // namespace Ilum