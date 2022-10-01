#include "Synchronization.hpp"
#include "Device.hpp"
#include "Backend/Vulkan/Device.hpp"
#include "Backend/Vulkan/Synchronization.hpp"

namespace Ilum::CUDA
{
HANDLE GetVkSemaphoreHandle(Vulkan::Device *device, Vulkan::Semaphore *semaphore, VkExternalSemaphoreHandleTypeFlagBitsKHR external_semaphore_handle_type_flag)
{
	HANDLE handle = {};

	VkSemaphoreGetWin32HandleInfoKHR handle_info = {};
	handle_info.sType                            = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.pNext                            = NULL;
	handle_info.semaphore                        = semaphore->GetHandle();
	handle_info.handleType                       = external_semaphore_handle_type_flag;

	device->GetSemaphoreWin32Handle(&handle_info, &handle);

	return handle;
}

Semaphore::Semaphore(Device *device, Vulkan::Device *vk_device, Vulkan::Semaphore *vk_semaphore) :
    RHISemaphore(device), p_device(device)
{
	cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc = {};

#ifdef _WIN64
	external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

	external_semaphore_handle_desc.handle.win32.handle = GetVkSemaphoreHandle(vk_device, vk_semaphore, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
	external_semaphore_handle_desc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
	external_semaphore_handle_desc.handle.fd = GetVkSemaphoreHandle(vk_device, vk_semaphore, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
#endif
	external_semaphore_handle_desc.flags = 0;

	cudaImportExternalSemaphore(&m_handle, &external_semaphore_handle_desc);
}

Semaphore::~Semaphore()
{
	cudaDestroyExternalSemaphore(m_handle);
}

void Semaphore::Signal()
{
	cudaExternalSemaphoreSignalParams params = {};

	params.params.fence.value = 0;
	params.flags              = 0;

	cudaSignalExternalSemaphoresAsync(&m_handle, &params, 1, p_device->GetSteam());
}

void Semaphore::Wait()
{
	cudaExternalSemaphoreWaitParams params = {};

	params.params.fence.value = 0;
	params.flags              = 0;

	cudaWaitExternalSemaphoresAsync(&m_handle, &params, 1, p_device->GetSteam());
}
}        // namespace Ilum::CUDA