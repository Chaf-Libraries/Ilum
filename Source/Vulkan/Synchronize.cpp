#include "Synchronize.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"

namespace Ilum::Vulkan
{
FencePool ::~FencePool()
{
	Wait();
	Reset();

	for (auto& fence : m_fences)
	{
		vkDestroyFence(RenderContext::GetDevice(), fence, nullptr);
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

	vkCreateFence(RenderContext::GetDevice(), &create_info, nullptr, &fence);

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

	vkWaitForFences(RenderContext::GetDevice(), m_active_fence_count, m_fences.data(), true, timeout);
}

void FencePool::Reset()
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkResetFences(RenderContext::GetDevice(), m_active_fence_count, m_fences.data());

	m_active_fence_count = 0;
}

SemaphorePool ::~SemaphorePool()
{
	Reset();

	for (auto& semaphore : m_semaphores)
	{
		vkDestroySemaphore(RenderContext::GetDevice(), semaphore, nullptr);
	}

	m_semaphores.clear();
}

VkSemaphore SemaphorePool::RequestSemaphore()
{
	std::lock_guard<std::mutex> lock(m_request_mutex);

	// Pop avaliable semaphore
	if (m_active_semaphore_count < m_semaphores.size())
	{
		return m_semaphores.at(m_active_semaphore_count++);
	}

		// Create new semaphore
	VkSemaphore semaphore = VK_NULL_HANDLE;

	VkSemaphoreCreateInfo create_info = {};
	create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	vkCreateSemaphore(RenderContext::GetDevice(), &create_info, nullptr, &semaphore);

	m_semaphores.push_back(semaphore);

	m_active_semaphore_count++;

	return m_semaphores.back();
}

VkSemaphore SemaphorePool::AllocateSemaphore()
{
	// Pop avaliable semaphore
	if (m_active_semaphore_count < m_semaphores.size())
	{
		VkSemaphore semaphore = m_semaphores.back();
		m_semaphores.pop_back();
		return semaphore;
	}

	// Create new semaphore
	VkSemaphore semaphore = VK_NULL_HANDLE;

	VkSemaphoreCreateInfo create_info = {};
	create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	vkCreateSemaphore(RenderContext::GetDevice(), &create_info, nullptr, &semaphore);
	return semaphore;
}

void SemaphorePool::ReleaseAllocatedSemaphore(VkSemaphore semaphore)
{
	std::lock_guard<std::mutex> lock(m_release_mutex);
	m_released_semaphores.push_back(semaphore);
}

void SemaphorePool::Reset()
{
	m_active_semaphore_count = 0;

	for (auto& semaphore : m_released_semaphores)
	{
		m_semaphores.push_back(semaphore);
	}

	m_released_semaphores.clear();
}
}        // namespace Ilum::Vulkan