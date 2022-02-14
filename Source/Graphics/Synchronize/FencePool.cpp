#include "FencePool.hpp"
#include "../Device/Device.hpp"

namespace Ilum::Graphics
{
FencePool::FencePool(const Device &device) :
    m_device(device)
{
}

FencePool::~FencePool()
{
	Wait();
	Reset();

	for (auto &fence : m_fences)
	{
		vkDestroyFence(m_device, fence, nullptr);
	}

	m_fences.clear();
}

VkFence &FencePool::RequestFence()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_active_fence_count < m_fences.size())
	{
		return m_fences.at(m_active_fence_count++);
	}

	VkFence fence = VK_NULL_HANDLE;

	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	vkCreateFence(m_device, &create_info, nullptr, &fence);

	m_fences.push_back(fence);
	m_active_fence_count++;

	return m_fences.back();
}

void FencePool::Wait(uint32_t timeout) const
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkWaitForFences(m_device, m_active_fence_count, m_fences.data(), true, timeout);
}

void FencePool::Reset()
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkResetFences(m_device, m_active_fence_count, m_fences.data());

	m_active_fence_count = 0;
}
}        // namespace Ilum::Graphics