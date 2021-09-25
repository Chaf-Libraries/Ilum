#include "CommandPool.hpp"

#include "Core/Device/LogicalDevice.hpp"

namespace Ilum
{
CommandPool::CommandPool(const LogicalDevice &logical_device, CommandPool::Usage usage, const std::thread::id &thread_id) :
    m_logical_device(logical_device), m_thread_id(thread_id), m_usage(usage)
{
	uint32_t queue_family = 0;
	switch (usage)
	{
		case Ilum::CommandPool::Usage::Graphics:
			queue_family = m_logical_device.getGraphicsFamily();
			break;
		case Ilum::CommandPool::Usage::Compute:
			queue_family = m_logical_device.getComputeFamily();
			break;
		case Ilum::CommandPool::Usage::Transfer:
			queue_family = m_logical_device.getTransferFamily();
			break;
		default:
			break;
	}

	VkCommandPoolCreateInfo command_pool_create_info = {};
	command_pool_create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_create_info.queueFamilyIndex        = queue_family;

	if (!VK_CHECK(vkCreateCommandPool(m_logical_device, &command_pool_create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create command pool");
		return;
	}
}

CommandPool::~CommandPool()
{
	if (m_handle)
	{
		vkDestroyCommandPool(m_logical_device, m_handle, nullptr);
	}
}

void CommandPool::reset()
{
	if (m_handle)
	{
		if (!VK_CHECK(vkResetCommandPool(m_logical_device, m_handle, 0)))
		{
			VK_ERROR("Failed to reset command pool!");
			return;
		}
	}
}

CommandPool::operator const VkCommandPool &() const
{
	return m_handle;
}

const LogicalDevice &CommandPool::getLogicalDevice() const
{
	return m_logical_device;
}

const VkCommandPool &CommandPool::getCommandPool() const
{
	return m_handle;
}

const std::thread::id &CommandPool::getThreadID() const
{
	return m_thread_id;
}

const VkQueue CommandPool::getQueue(uint32_t index) const
{
	const std::vector<VkQueue> *queues = nullptr;
	switch (m_usage)
	{
		case Ilum::CommandPool::Usage::Graphics:
			queues = &m_logical_device.getGraphicsQueues();
			return queues->at(index % queues->size());
		case Ilum::CommandPool::Usage::Compute:
			queues = &m_logical_device.getComputeQueues();
			return queues->at(index % queues->size());
		case Ilum::CommandPool::Usage::Transfer:
			queues = &m_logical_device.getTransferQueues();
			return queues->at(index % queues->size());
		default:
			return VK_NULL_HANDLE;
	}
	return VK_NULL_HANDLE;
}
}        // namespace Ilum