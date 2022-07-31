#pragma once

#include "Command.hpp"
#include "SyncAllocator.hpp"

#include <map>

namespace Ilum
{
class RHIDevice;
class FenceAllocator;
class SemaphoreAllocator;

class Frame
{
  public:
	Frame(RHIDevice *device);
	~Frame() = default;

	void Reset();

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, VkQueueFlagBits queue = VK_QUEUE_GRAPHICS_BIT, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	CommandPool   &RequestCommandPool(VkQueueFlagBits queue, CommandPool::ResetMode reset_mode = CommandPool::ResetMode::ResetPool);
	VkFence       &RequestFence();
	VkSemaphore    RequestSemaphore();
	VkSemaphore    AllocateSemaphore();
	void           ReleaseAllocatedSemaphore(VkSemaphore semaphore);

  private:
	RHIDevice *p_device=nullptr;

	std::map<size_t, std::unique_ptr<CommandPool>> m_command_pools;
	std::unique_ptr<FenceAllocator>                m_fence_allocator     = nullptr;
	std::unique_ptr<SemaphoreAllocator>            m_semaphore_allocator = nullptr;

	uint32_t                 m_current_profile = 0;
};

}        // namespace Ilum