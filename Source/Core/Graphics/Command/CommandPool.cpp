#include "CommandPool.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Engine/Context.hpp"
#include "Core/Engine/Engine.hpp"
#include "Core/Graphics/GraphicsContext.hpp"

namespace Ilum
{
CommandPool::CommandPool(VkQueueFlagBits queue_type, const std::thread::id &thread_id) :
    m_logical_device(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice()), m_thread_id(thread_id), m_queue_type(queue_type)
{
	uint32_t queue_family = 0;

	if (queue_type & VK_QUEUE_GRAPHICS_BIT)
	{
		queue_family = m_logical_device.getGraphicsFamily();
	}
	else if (queue_type & VK_QUEUE_COMPUTE_BIT)
	{
		queue_family = m_logical_device.getGraphicsFamily();
	}
	else if (queue_type & VK_QUEUE_TRANSFER_BIT)
	{
		queue_family = m_logical_device.getTransferFamily();
	}
	else
	{
		queue_family = m_logical_device.getGraphicsFamily();
	}

	VkCommandPoolCreateInfo command_pool_create_info = {};
	command_pool_create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_create_info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
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
	if (m_queue_type & VK_QUEUE_GRAPHICS_BIT)
	{
		return m_logical_device.getGraphicsQueues().at(index % m_logical_device.getGraphicsQueues().size());
	}
	else if (m_queue_type & VK_QUEUE_COMPUTE_BIT)
	{
		return m_logical_device.getComputeQueues().at(index % m_logical_device.getComputeQueues().size());
	}
	else if (m_queue_type & VK_QUEUE_TRANSFER_BIT)
	{
		return m_logical_device.getTransferQueues().at(index % m_logical_device.getTransferQueues().size());
	}

	return VK_NULL_HANDLE;
}
}        // namespace Ilum