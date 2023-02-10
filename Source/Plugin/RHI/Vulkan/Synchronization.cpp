#include "Synchronization.hpp"
#include "Device.hpp"

#include <dxgi1_2.h>

namespace Ilum::Vulkan
{
Fence::Fence(RHIDevice *device) :
    RHIFence(device)
{
	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	vkCreateFence(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
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

Semaphore::Semaphore(RHIDevice *device) :
    RHISemaphore(device)
{
	VkSemaphoreCreateInfo create_info = {};
	create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	create_info.pNext                 = nullptr;

#ifdef CUDA_ENABLE
#	if defined(_WIN64)
	WindowsSecurityAttributes windows_security_attributes;

	VkExportSemaphoreWin32HandleInfoKHR export_semaphore_win32_handle_info = {};
	export_semaphore_win32_handle_info.sType                               = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
	export_semaphore_win32_handle_info.pNext                               = NULL;
	export_semaphore_win32_handle_info.pAttributes                         = &windows_security_attributes;
	export_semaphore_win32_handle_info.dwAccess                            = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	export_semaphore_win32_handle_info.name                                = (LPCWSTR) NULL;
#	endif
	VkExportSemaphoreCreateInfoKHR export_semaphore_create_info = {};

	export_semaphore_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#	if defined(_WIN64)
	export_semaphore_create_info.pNext       = &export_semaphore_win32_handle_info;
	export_semaphore_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#	else
	export_semaphore_create_info.pNext       = NULL;
	export_semaphore_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#	endif
	create_info.pNext = &export_semaphore_create_info;
#endif        // CUDA_ENABLE



	vkCreateSemaphore(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
}

Semaphore::~Semaphore()
{
	if (m_handle)
	{
		vkDestroySemaphore(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
	}
}

void Semaphore::SetName(const std::string &name)
{
	VkDebugUtilsObjectNameInfoEXT info = {};
	info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	info.pObjectName                   = name.c_str();
	info.objectHandle                  = (uint64_t) m_handle;
	info.objectType                    = VK_OBJECT_TYPE_SEMAPHORE;
	static_cast<Device *>(p_device)->SetVulkanObjectName(info);
}

VkSemaphore Semaphore::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan