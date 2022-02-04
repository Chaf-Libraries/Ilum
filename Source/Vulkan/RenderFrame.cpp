#include "RenderFrame.hpp"
#include "Synchronize.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Vulkan
{
RenderFrame::RenderFrame()
{
	m_fence_pool = std::make_unique<FencePool>();
	m_semaphore_pool = std::make_unique<SemaphorePool>();
}

RenderFrame::~RenderFrame()
{
}

void RenderFrame::Reset()
{
	for (auto& [pool_index, cmd_pool] : m_command_pools)
	{
		cmd_pool->Reset();
	}
}

CommandBuffer &RenderFrame::RequestCommandBuffer(VkCommandBufferLevel level, QueueFamily queue, CommandPool::ResetMode reset_mode)
{
	auto &pool = RequestCommandPool(queue, reset_mode);
	return pool.RequestCommandBuffer(level);
}

VkFence &RenderFrame::RequestFence()
{
	return m_fence_pool->RequestFence();
}

const FencePool &RenderFrame::GetFencePool() const
{
	return *m_fence_pool;
}

VkSemaphore RenderFrame::RequestSemaphore()
{
	return m_semaphore_pool->RequestSemaphore();
}

VkSemaphore RenderFrame::AllocateSemaphore()
{
	return m_semaphore_pool->AllocateSemaphore();
}

void RenderFrame::ReleaseAllocatedSemaphore(VkSemaphore semaphore)
{
	m_semaphore_pool->ReleaseAllocatedSemaphore(semaphore);
}

const SemaphorePool &RenderFrame::GetSemaphorePool() const
{
	return *m_semaphore_pool;
}

CommandPool &RenderFrame::RequestCommandPool(QueueFamily queue, CommandPool::ResetMode reset_mode)
{
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(queue));
	Core::HashCombine(hash, static_cast<size_t>(reset_mode));
	Core::HashCombine(hash, thread_id);

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		m_command_pools.emplace(hash, std::make_unique<CommandPool>(queue, reset_mode, thread_id));
	}

	return *m_command_pools[hash];
}
}        // namespace Ilum::Vulkan