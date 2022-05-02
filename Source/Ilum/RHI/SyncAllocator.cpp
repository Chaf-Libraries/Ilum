#include "SyncAllocator.hpp"
#include "Device.hpp"

namespace Ilum
{
FenceAllocator::FenceAllocator(RHIDevice *device) :
    p_device(device)
{
}

FenceAllocator::~FenceAllocator()
{
	Wait();
	Reset();

	for (auto &fence : m_fences)
	{
		vkDestroyFence(p_device->m_device, fence, nullptr);
	}

	m_fences.clear();
}

VkFence &FenceAllocator::RequestFence()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_active_fence_count < m_fences.size())
	{
		return m_fences.at(m_active_fence_count++);
	}

	VkFence fence = VK_NULL_HANDLE;

	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	vkCreateFence(p_device->m_device, &create_info, nullptr, &fence);

	m_fences.push_back(fence);
	m_active_fence_count++;

	return m_fences.back();
}

void FenceAllocator::Wait(uint32_t timeout) const
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkWaitForFences(p_device->m_device, m_active_fence_count, m_fences.data(), true, timeout);
}

void FenceAllocator::Reset()
{
	if (m_active_fence_count < 1 || m_fences.empty())
	{
		return;
	}

	vkResetFences(p_device->m_device, m_active_fence_count, m_fences.data());

	m_active_fence_count = 0;
}

SemaphoreAllocator::SemaphoreAllocator(RHIDevice *device):
    p_device(device)
{
}

SemaphoreAllocator::~SemaphoreAllocator()
{
	Reset();

	for (auto &semaphore : m_semaphores)
	{
		vkDestroySemaphore(p_device->m_device, semaphore, nullptr);
	}

	m_semaphores.clear();
}

VkSemaphore SemaphoreAllocator::RequestSemaphore()
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

	vkCreateSemaphore(p_device->m_device, &create_info, nullptr, &semaphore);

	m_semaphores.push_back(semaphore);

	m_active_semaphore_count++;

	return m_semaphores.back();
}

VkSemaphore SemaphoreAllocator::AllocateSemaphore()
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

	vkCreateSemaphore(p_device->m_device, &create_info, nullptr, &semaphore);
	return semaphore;
}

void SemaphoreAllocator::ReleaseAllocatedSemaphore(VkSemaphore semaphore)
{
	std::lock_guard<std::mutex> lock(m_release_mutex);
	m_released_semaphores.push_back(semaphore);
}

void SemaphoreAllocator::Reset()
{
	m_active_semaphore_count = 0;

	for (auto &semaphore : m_released_semaphores)
	{
		m_semaphores.push_back(semaphore);
	}

	m_released_semaphores.clear();
}
}        // namespace Ilum