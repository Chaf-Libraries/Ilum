#include "Synchronization.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
Fence::Fence(RHIDevice *device) :
    RHIFence(device)
{
	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	vkCreateFence(static_cast<Device*>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
}

Fence::~Fence()
{
	if (m_handle)
	{
		vkDestroyFence(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
	}
}

void Fence::Wait(uint64_t timeout)
{
	vkWaitForFences(static_cast<Device *>(p_device)->GetDevice(), 1, &m_handle, true, timeout);
}

void Fence::Reset()
{
	vkResetFences(static_cast<Device *>(p_device)->GetDevice(), 1, &m_handle);
}

VkFence Fence::GetHandle() const
{
	return m_handle;
}

Semaphore::Semaphore(RHIDevice *device):
    RHISemaphore(device)
{
	VkSemaphoreCreateInfo create_info = {};
	create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	vkCreateSemaphore(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
}

Semaphore::~Semaphore()
{
	if (m_handle)
	{
		vkDestroySemaphore(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
	}
}

VkSemaphore Semaphore::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan