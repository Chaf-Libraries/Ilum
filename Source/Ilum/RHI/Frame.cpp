#include "Frame.hpp"
#include "Device.hpp"

#include <Core/Hash.hpp>

namespace Ilum
{
Frame::Frame(RHIDevice *device) :
    p_device(device)
{
	m_fence_allocator     = std::make_unique<FenceAllocator>(p_device);
	m_semaphore_allocator = std::make_unique<SemaphoreAllocator>(p_device);
}

void Frame::Reset()
{
	m_fence_allocator->Wait();
	m_fence_allocator->Reset();
	m_semaphore_allocator->Reset();

	for (auto &[pool_index, cmd_pool] : m_command_pools)
	{
		cmd_pool->Reset();
	}
}

CommandBuffer &Frame::RequestCommandBuffer(VkCommandBufferLevel level, VkQueueFlagBits queue, CommandPool::ResetMode reset_mode)
{
	auto &pool = RequestCommandPool(queue, reset_mode);
	return pool.RequestCommandBuffer(level);
}

CommandPool &Frame::RequestCommandPool(VkQueueFlagBits queue, CommandPool::ResetMode reset_mode)
{
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	HashCombine(hash, queue);
	HashCombine(hash, reset_mode);
	HashCombine(hash, thread_id);

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		m_command_pools.emplace(hash, std::make_unique<CommandPool>(p_device, queue, reset_mode, thread_id));
	}

	return *m_command_pools[hash];
}

VkFence &Frame::RequestFence()
{
	return m_fence_allocator->RequestFence();
}

VkSemaphore Frame::RequestSemaphore()
{
	return m_semaphore_allocator->RequestSemaphore();
}

VkSemaphore Frame::AllocateSemaphore()
{
	return m_semaphore_allocator->AllocateSemaphore();
}

void Frame::ReleaseAllocatedSemaphore(VkSemaphore semaphore)
{
	m_semaphore_allocator->ReleaseAllocatedSemaphore(semaphore);
}

}        // namespace Ilum