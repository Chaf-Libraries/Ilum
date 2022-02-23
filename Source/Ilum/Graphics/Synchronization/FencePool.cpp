#include "FencePool.hpp"

#include "Device/LogicalDevice.hpp"
#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
FencePool::~FencePool()
{
	wait();
	reset();

	for (auto &fence : m_fences)
	{
		vkDestroyFence(GraphicsContext::instance()->getLogicalDevice(), fence, nullptr);
	}

	m_fences.clear();
}

VkFence &FencePool::requestFence()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_active_fence_count < m_fences.size())
	{
		return m_fences.at(m_active_fence_count++);
	}

	VkFence fence = VK_NULL_HANDLE;

	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	vkCreateFence(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &fence);

	m_fences.push_back(fence);
	m_active_fence_count++;

	return m_fences.back();
}

void FencePool::wait(uint32_t timeout) const
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkWaitForFences(GraphicsContext::instance()->getLogicalDevice(), m_active_fence_count, m_fences.data(), true, timeout);
}

void FencePool::reset()
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkResetFences(GraphicsContext::instance()->getLogicalDevice(), m_active_fence_count, m_fences.data());

	m_active_fence_count = 0;
}
}        // namespace Ilum::Graphics