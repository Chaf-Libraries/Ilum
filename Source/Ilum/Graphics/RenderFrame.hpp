#pragma once

#include "Command/CommandBuffer.hpp"
#include "Command/CommandPool.hpp"

#include <map>

namespace Ilum
{
class FencePool;
class SemaphorePool;

class RenderFrame
{
  public:
	RenderFrame();
	~RenderFrame() = default;

	void reset();

	CommandBuffer &requestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, QueueUsage queue = QueueUsage::Graphics, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	CommandPool &  requestCommandPool(QueueUsage queue, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	VkFence &      requestFence();
	VkSemaphore    requestSemaphore();
	VkSemaphore    allocateSemaphore();
	void           releaseAllocatedSemaphore(VkSemaphore semaphore);

  private:
	std::map<size_t, scope<CommandPool>> m_command_pools;
	scope<FencePool>                     m_fence_pool     = nullptr;
	scope<SemaphorePool>                 m_semaphore_pool = nullptr;
	std::mutex                           cmd_pool_mutex;
};
}        // namespace Ilum::Graphics