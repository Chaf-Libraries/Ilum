#include "Queue.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/Vulkan.hpp"
#include "Graphics/RenderFrame.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
Queue::Queue(VkQueue handle) :
    m_handle(handle)
{
}

void Queue::submit(const CommandBuffer &command_buffer,
                   const VkSemaphore &  signal_semaphore,
                   const VkSemaphore &  wait_semaphore,
                   const VkFence &      fence,
                   VkPipelineStageFlags wait_stages)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command_buffer.getCommandBuffer();

	submit_info.pWaitDstStageMask = &wait_stages;

	if (wait_semaphore)
	{
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores    = &wait_semaphore;
	}

	if (signal_semaphore)
	{
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &signal_semaphore;
	}

	if (!VK_CHECK(vkQueueSubmit(m_handle, 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}
}

void Queue::submit(const std::vector<VkCommandBuffer> &command_buffers, 
	const VkSemaphore &signal_semaphore, 
	const VkSemaphore &wait_semaphore, 
	const VkFence &fence, 
	VkPipelineStageFlags wait_stages)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());
	submit_info.pCommandBuffers    = command_buffers.data();

	submit_info.pWaitDstStageMask = &wait_stages;

	if (wait_semaphore)
	{
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores    = &wait_semaphore;
	}

	if (signal_semaphore)
	{
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &signal_semaphore;
	}

	if (!VK_CHECK(vkQueueSubmit(m_handle, 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}
}

void Queue::submit(const CommandBuffer &                    command_buffer,
                   const std::vector<VkSemaphore> &         signal_semaphores,
                   const std::vector<VkSemaphore> &         wait_semaphores,
                   const VkFence &                          fence,
                   const std::vector<VkPipelineStageFlags> &wait_stages)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command_buffer.getCommandBuffer();

	submit_info.pWaitDstStageMask = wait_stages.data();

	if (!wait_semaphores.empty())
	{
		submit_info.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
		submit_info.pWaitSemaphores    = wait_semaphores.data();
	}

	if (!signal_semaphores.empty())
	{
		submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
		submit_info.pSignalSemaphores    = signal_semaphores.data();
	}

	if (!VK_CHECK(vkQueueSubmit(m_handle, 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}
}

void Queue::submit(const CommandBuffer &command_buffer, const SubmitInfo &submit_info)
{
	submit(command_buffer, {submit_info.signal_semaphore}, submit_info.wait_semaphores, submit_info.fence, submit_info.wait_stages);
}

void Queue::submitIdle(const CommandBuffer &command_buffer)
{
	waitIdle();

	std::lock_guard<std::mutex> lock(m_mutex);

	VkSubmitInfo submit_info       = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command_buffer.getCommandBuffer();

	VkFence fence = VK_NULL_HANDLE;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	vkCreateFence(GraphicsContext::instance()->getLogicalDevice(), &fence_create_info, nullptr, &fence);

	vkResetFences(GraphicsContext::instance()->getLogicalDevice(), 1, &fence);

	if (!VK_CHECK(vkQueueSubmit(m_handle, 1, &submit_info, fence)))
	{
		VK_ERROR("Failed to submit queue!");
		return;
	}

	vkWaitForFences(GraphicsContext::instance()->getLogicalDevice(), 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkDestroyFence(GraphicsContext::instance()->getLogicalDevice(), fence, nullptr);
}

void Queue::waitIdle()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	vkQueueWaitIdle(m_handle);
}

const VkQueue &Queue::getQueue() const
{
	return m_handle;
}

Queue::operator const VkQueue &() const
{
	return m_handle;
}
}        // namespace Ilum