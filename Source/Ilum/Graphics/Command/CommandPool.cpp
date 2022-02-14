#include "CommandPool.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
CommandPool::CommandPool(QueueUsage usage, const std::thread::id &thread_id) :
    m_thread_id(thread_id), m_queue_usage(usage)
{
	uint32_t queue_family = 0;

	switch (m_queue_usage)
	{
		case Ilum::QueueUsage::Graphics:
			queue_family = Graphics::RenderContext::GetDevice().GetQueueFamily(Graphics::QueueFamily::Graphics);
			break;
		case Ilum::QueueUsage::Compute:
			queue_family = Graphics::RenderContext::GetDevice().GetQueueFamily(Graphics::QueueFamily::Compute);
			break;
		case Ilum::QueueUsage::Transfer:
			queue_family = Graphics::RenderContext::GetDevice().GetQueueFamily(Graphics::QueueFamily::Transfer);
			break;
		case Ilum::QueueUsage::Present:
			queue_family = Graphics::RenderContext::GetDevice().GetQueueFamily(Graphics::QueueFamily::Present);
			break;
		default:
			break;
	}

	VkCommandPoolCreateInfo command_pool_create_info = {};
	command_pool_create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_create_info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	command_pool_create_info.queueFamilyIndex        = queue_family;

	if (!VK_CHECK(vkCreateCommandPool(Graphics::RenderContext::GetDevice(), &command_pool_create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create command pool");
		return;
	}
}

CommandPool::~CommandPool()
{
	if (m_handle)
	{
		vkDestroyCommandPool(Graphics::RenderContext::GetDevice(), m_handle, nullptr);
	}
}

void CommandPool::reset()
{
	if (m_handle)
	{
		if (!VK_CHECK(vkResetCommandPool(Graphics::RenderContext::GetDevice(), m_handle, 0)))
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

QueueUsage CommandPool::getUsage() const
{
	return m_queue_usage;
}
}        // namespace Ilum