#include "SemaphorePool.hpp"
#include "../Device/Device.hpp"

namespace Ilum::Graphics
{
SemaphorePool::SemaphorePool(const Device &device):
    m_device(device)
{
}

SemaphorePool ::~SemaphorePool()
{
	Reset();

	for (auto &semaphore : m_semaphores)
	{
		vkDestroySemaphore(m_device, semaphore, nullptr);
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

	vkCreateSemaphore(m_device, &create_info, nullptr, &semaphore);

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

	vkCreateSemaphore(m_device, &create_info, nullptr, &semaphore);
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

	for (auto &semaphore : m_released_semaphores)
	{
		m_semaphores.push_back(semaphore);
	}

	m_released_semaphores.clear();
}
}