#pragma once

#include "Command/CommandBuffer.hpp"
#include "Command/CommandPool.hpp"

#include <map>

namespace Ilum::Graphics
{
class FencePool;
class SemaphorePool;
class Device;

class RenderFrame
{
  public:
	RenderFrame(const Device &device);
	~RenderFrame() = default;

	void Reset();

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, QueueFamily queue = QueueFamily::Graphics, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	CommandPool &  RequestCommandPool(QueueFamily queue, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	VkFence &      RequestFence();
	VkSemaphore    RequestSemaphore();
	VkSemaphore    AllocateSemaphore();
	void           ReleaseAllocatedSemaphore(VkSemaphore semaphore);

  private:
	const Device &m_device;

	std::map<size_t, std::unique_ptr<CommandPool>> m_command_pools;
	std::unique_ptr<FencePool>                     m_fence_pool     = nullptr;
	std::unique_ptr<SemaphorePool>                 m_semaphore_pool = nullptr;
};
}        // namespace Ilum::Graphics