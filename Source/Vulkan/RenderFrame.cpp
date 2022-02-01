#include "RenderFrame.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Vulkan
{
RenderFrame::RenderFrame()
{
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