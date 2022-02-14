#include "RenderFrame.hpp"
#include "Device/Device.hpp"
#include "Synchronize/FencePool.hpp"
#include "Synchronize/SemaphorePool.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Graphics
{
RenderFrame::RenderFrame(const Device &device) :
    m_device(device)
{
	m_fence_pool     = std::make_unique<FencePool>(m_device);
	m_semaphore_pool = std::make_unique<SemaphorePool>(m_device);
}

void RenderFrame::Reset()
{
	m_fence_pool->Wait();
	m_fence_pool->Reset();
	m_semaphore_pool->Reset();

	for (auto &[pool_index, cmd_pool] : m_command_pools)
	{
		cmd_pool->Reset();
	}
}

CommandBuffer& RenderFrame::RequestCommandBuffer(VkCommandBufferLevel level, QueueFamily queue, CommandPool::ResetMode reset_mode)
{
	auto &pool = RequestCommandPool(queue, reset_mode);
	return pool.RequestCommandBuffer(level);
}

CommandPool& RenderFrame::RequestCommandPool(QueueFamily queue, CommandPool::ResetMode reset_mode)
{
	auto thread_id = std::this_thread::get_id();

	size_t hash = 0;
	Core::HashCombine(hash, static_cast<size_t>(queue));
	Core::HashCombine(hash, static_cast<size_t>(reset_mode));
	Core::HashCombine(hash, thread_id);

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		m_command_pools.emplace(hash, std::make_unique<CommandPool>(m_device, queue, reset_mode, thread_id));
	}

	return *m_command_pools[hash];
}

VkFence& RenderFrame::RequestFence()
{
	return m_fence_pool->RequestFence();
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
}        // namespace Ilum::Graphics