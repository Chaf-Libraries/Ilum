#pragma once

#include "Command.hpp"
#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class FencePool;
class SemaphorePool;

class RenderFrame
{
  public:
	RenderFrame();

	~RenderFrame();

	void Reset();

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, QueueFamily queue = QueueFamily::Graphics, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);

	VkFence &RequestFence();

	const FencePool &GetFencePool() const;

	VkSemaphore RequestSemaphore();

	VkSemaphore AllocateSemaphore();

	void ReleaseAllocatedSemaphore(VkSemaphore semaphore);

	const SemaphorePool &GetSemaphorePool() const;

  private:
	CommandPool &RequestCommandPool(QueueFamily queue, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);

  private:
	std::unordered_map<size_t, std::unique_ptr<CommandPool>> m_command_pools;
	std::unique_ptr<FencePool>                               m_fence_pool;
	std::unique_ptr<SemaphorePool>                           m_semaphore_pool;
};
}        // namespace Ilum::Vulkan