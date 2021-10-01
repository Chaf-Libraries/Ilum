#include "CommandBuffer.hpp"
#include "CommandPool.hpp"

#include "Core/Device/LogicalDevice.hpp"

namespace Ilum
{
CommandBuffer::CommandBuffer(const CommandPool &command_pool, VkQueueFlagBits queue_type, VkCommandBufferLevel level) :
    m_command_pool(command_pool), m_queue_type(queue_type), m_logical_device(command_pool.getLogicalDevice())
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.commandPool                 = command_pool;
	command_buffer_allocate_info.level                       = level;
	command_buffer_allocate_info.commandBufferCount          = 1;
	if (!VK_CHECK(vkAllocateCommandBuffers(m_logical_device, &command_buffer_allocate_info, &m_handle)))
	{
		VK_ERROR("Failed to create command buffer!");
		return;
	}
}

CommandBuffer::~CommandBuffer()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(m_logical_device, m_command_pool, 1, &m_handle);
	}
}

void CommandBuffer::reset()
{
	m_state = State::Initial;

	if (!VK_CHECK(vkResetCommandBuffer(m_handle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)))
	{
		VK_ERROR("Failed to reset command buffer!");
		return;
	}
}

bool CommandBuffer::begin(VkCommandBufferUsageFlagBits usage)
{
	if (m_state == State::Recording)
	{
		VK_WARN("Command buffer is already recording");
		return false;
	}

	VkCommandBufferBeginInfo command_buffer_begin_info = {};
	command_buffer_begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	command_buffer_begin_info.flags                    = usage;

	if (!VK_CHECK(vkBeginCommandBuffer(m_handle, &command_buffer_begin_info)))
	{
		VK_ERROR("Failed to begin command buffer!");
		return false;
	}

	m_state = State::Recording;
	return true;
}

void CommandBuffer::end()
{
	if (m_state != State::Recording)
	{
		VK_WARN("Command buffer is not recording!");
		return;
	}

	if (!VK_CHECK(vkEndCommandBuffer(m_handle)))
	{
		VK_ERROR("Failed to end command buffer!");
		m_state = State::Invalid;
		return;
	}

	m_state = State::Executable;
}

void CommandBuffer::submitIdle(uint32_t queue_index)
{
	if (m_state != State::Executable)
	{
		end();
	}

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_handle;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	VkFence fence;

	if (!VK_CHECK(vkCreateFence(m_logical_device, &fence_create_info, nullptr, &fence)))
	{
		VK_ERROR("Failed to create fence!");
		return;
	}

	if (!VK_CHECK(vkResetFences(m_logical_device, 1, &fence)))
	{
		VK_ERROR("Failed to reset fence!");
		return;
	}

	if (!VK_CHECK(vkQueueSubmit(m_command_pool.getQueue(queue_index), 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}

	if (!VK_CHECK(vkWaitForFences(m_logical_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max())))
	{
		VK_ERROR("Failed to wait for fence!");
		return;
	}

	vkDestroyFence(m_logical_device, fence, nullptr);
}

void CommandBuffer::submit(const VkSemaphore &wait_semaphore, const VkSemaphore &signal_semaphore, VkFence fence, VkShaderStageFlags wait_stages, uint32_t queue_index)
{
	if (m_state != State::Executable)
	{
		VK_ERROR("Command buffer is not executable!");
		return;
	}

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_handle;

	if (wait_semaphore != VK_NULL_HANDLE)
	{
		submit_info.pWaitDstStageMask  = &wait_stages;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores    = &wait_semaphore;
	}

	if (signal_semaphore != VK_NULL_HANDLE)
	{
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &signal_semaphore;
	}

	if (fence != VK_NULL_HANDLE)
	{
		vkResetFences(m_logical_device, 1, &fence);
	}

	if (!VK_CHECK(vkQueueSubmit(m_command_pool.getQueue(queue_index), 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}
}

void CommandBuffer::submit(const std::vector<VkSemaphore> &wait_semaphores, const std::vector<VkSemaphore> &signal_semaphores, VkFence fence, const std::vector<VkShaderStageFlags> &wait_stages, uint32_t queue_index)
{
	if (m_state != State::Executable)
	{
		VK_ERROR("Command buffer is not executable!");
		return;
	}

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_handle;

	if (!wait_semaphores.empty())
	{
		submit_info.pWaitDstStageMask  = wait_stages.data();
		submit_info.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
		submit_info.pWaitSemaphores    = wait_semaphores.data();
	}

	if (!signal_semaphores.empty())
	{
		submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
		submit_info.pSignalSemaphores    = signal_semaphores.data();
	}

	if (fence != VK_NULL_HANDLE)
	{
		vkResetFences(m_logical_device, 1, &fence);
	}

	if (!VK_CHECK(vkQueueSubmit(m_command_pool.getQueue(queue_index), 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}
}

const CommandPool &CommandBuffer::getCommandPool() const
{
	return m_command_pool;
}

CommandBuffer::operator const VkCommandBuffer &() const
{
	return m_handle;
}

const VkCommandBuffer &CommandBuffer::getCommandBuffer() const
{
	return m_handle;
}

const CommandBuffer::State &CommandBuffer::getState() const
{
	return m_state;
}
}        // namespace Ilum