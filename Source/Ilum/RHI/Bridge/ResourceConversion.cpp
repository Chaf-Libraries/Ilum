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

	if (!texture->GetDesc().external)
	{
		LOG_ERROR("Texture {} is not external accessable!", texture->GetDesc().name);
		return handle;
	}

	VkMemoryGetWin32HandleInfoKHR handle_info = {};
	handle_info.sType                         = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.pNext                         = NULL;
	handle_info.memory                        = texture->GetMemory();
	handle_info.handleType                    = (VkExternalMemoryHandleTypeFlagBitsKHR) external_memory_handle_type_flag;

	device->GetMemoryWin32Handle(&handle_info, &handle);

	return handle;
}

inline std::unique_ptr<RHITexture> MapTextureVulkanToCUDA(Vulkan::Device *device, Vulkan::Texture *texture)
{
	cudaExternalMemoryHandleDesc cuda_external_memory_handle_desc = {};

#ifdef _WIN64
	cuda_external_memory_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
	cuda_external_memory_handle_desc.handle.win32.handle = GetVkImageMemHandle(device, texture, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
	cuda_external_memory_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
	cuda_external_memory_handle_desc.handle.fd = GetVkImageMemHandle(device, texture, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
	cuda_external_memory_handle_desc.size = texture->GetMemorySize();

	cudaExternalMemory_t cuda_external_memory = {};
	cudaImportExternalMemory(&cuda_external_memory, &cuda_external_memory_handle_desc);

	cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;

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
			//MapTextureVulkanToCUDA(texture);
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