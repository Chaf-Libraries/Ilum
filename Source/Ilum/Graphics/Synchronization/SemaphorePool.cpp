#include "SemaphorePool.hpp"

#include "Device/LogicalDevice.hpp"
#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
SemaphorePool ::~SemaphorePool()
{
	reset();

	for (auto &semaphore : m_semaphores)
	{
		vkDestroySemaphore(GraphicsContext::instance()->getLogicalDevice(), semaphore, nullptr);
	}

	m_semaphores.clear();
}

VkSemaphore SemaphorePool::requestSemaphore()
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

	vkCreateSemaphore(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &semaphore);

	m_semaphores.push_back(semaphore);

	m_active_semaphore_count++;

	return m_semaphores.back();
}

VkSemaphore SemaphorePool::allocateSemaphore()
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

	vkCreateSemaphore(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &semaphore);
	return semaphore;
}

void SemaphorePool::releaseAllocatedSemaphore(VkSemaphore semaphore)
{
	std::lock_guard<std::mutex> lock(m_release_mutex);
	m_released_semaphores.push_back(semaphore);
}

void SemaphorePool::reset()
{
	m_active_semaphore_count = 0;

	for (auto &semaphore : m_released_semaphores)
	{
		m_semaphores.push_back(semaphore);
	}

	m_released_semaphores.clear();
}
}