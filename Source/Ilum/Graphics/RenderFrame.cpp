#include "RenderFrame.hpp"

#include "Synchronization/FencePool.hpp"
#include "Synchronization/SemaphorePool.hpp"

#include "Command/CommandBuffer.hpp"
#include "Command/CommandPool.hpp"

#include "Utils/Hash.hpp"

namespace Ilum
{
RenderFrame::RenderFrame()
{
	m_fence_pool     = createScope<FencePool>();
	m_semaphore_pool = createScope<SemaphorePool>();
}

void RenderFrame::reset()
{
	m_fence_pool->wait();
	m_fence_pool->reset();
	m_semaphore_pool->reset();

	for (auto &[pool_index, cmd_pool] : m_command_pools)
	{
		cmd_pool->reset();
	}
}

CommandBuffer &RenderFrame::requestCommandBuffer(VkCommandBufferLevel level, QueueUsage queue, CommandPool::ResetMode reset_mode)
{
	auto &pool = requestCommandPool(queue, reset_mode);
	return pool.requestCommandBuffer(level);
}

CommandPool &RenderFrame::requestCommandPool(QueueUsage queue, CommandPool::ResetMode reset_mode)
{
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	hash_combine(hash, static_cast<size_t>(queue));
	hash_combine(hash, static_cast<size_t>(reset_mode));
	hash_combine(hash, thread_id);

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		std::lock_guard<std::mutex> lock(cmd_pool_mutex);
		m_command_pools.emplace(hash, createScope<CommandPool>(queue, reset_mode, thread_id));
	}

	return *m_command_pools[hash];
}

VkFence &RenderFrame::requestFence()
{
	return m_fence_pool->requestFence();
}

VkSemaphore RenderFrame::requestSemaphore()
{
	return m_semaphore_pool->requestSemaphore();
}

VkSemaphore RenderFrame::allocateSemaphore()
{
	return m_semaphore_pool->allocateSemaphore();
}

void RenderFrame::releaseAllocatedSemaphore(VkSemaphore semaphore)
{
	m_semaphore_pool->releaseAllocatedSemaphore(semaphore);
}
}        // namespace Ilum::Graphics