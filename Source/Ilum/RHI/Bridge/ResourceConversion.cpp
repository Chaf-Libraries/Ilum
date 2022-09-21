#include "ResourceConversion.hpp"
#include "Backend/Vulkan/Device.hpp"
#include "Backend/Vulkan/Texture.hpp"
#include "RHIContext.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN64
#	include <Windows.h>
#endif        // _WIN64

#include <volk.h>

namespace Ilum
{
HANDLE GetVkImageMemHandle(Vulkan::Device *device, Vulkan::Texture *texture, VkExternalMemoryHandleTypeFlagsKHR external_memory_handle_type_flag)
{
	HANDLE handle = {};

	VkMemoryGetWin32HandleInfoKHR handle_info = {};
	handle_info.sType                         = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.pNext                         = NULL;
	// handle_info.memory                        = textureImageMemory;
	// handle_info.handleType                    = (VkExternalMemoryHandleTypeFlagBitsKHR) external_memory_handle_type_flag;
	// vma
	device->GetMemoryWin32Handle(&handle_info, &handle);
	return handle;
}

inline std::unique_ptr<RHITexture> MapTextureVulkanToCUDA(RHITexture *texture)
{
	cudaExternalMemoryHandleDesc cuda_external_memory_handle_desc = {};
#ifdef _WIN64
	cuda_external_memory_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
	//cuda_external_memory_handle_desc.handle.win32.handle = GetVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

	cudaExtMemHandleDesc.handle.fd =
	    getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif

	return nullptr;
}

inline std::unique_ptr<RHIBuffer> MapBufferVulkanToCUDA(RHIBuffer *buffer)
{
	return nullptr;
}

std::unique_ptr<RHITexture> MapTextureToCUDA(RHITexture *texture)
{
	switch (texture->GetBackend())
	{
		case RHIBackend::Vulkan:
			MapTextureVulkanToCUDA(texture);
			break;
		default:
			break;
	}
	return nullptr;
}

std::unique_ptr<RHIBuffer> MapBufferToCUDA(RHIBuffer *buffer)
{
	switch (buffer->GetBackend())
	{
		case RHIBackend::Vulkan:
			MapBufferVulkanToCUDA(buffer);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum