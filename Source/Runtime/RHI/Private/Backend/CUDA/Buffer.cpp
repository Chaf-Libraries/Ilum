#include "Buffer.hpp"
#include "Backend/Vulkan/Buffer.hpp"
#include "Backend/Vulkan/Device.hpp"

namespace Ilum::CUDA
{
HANDLE GetVkBufferMemHandle(Vulkan::Device *device, Vulkan::Buffer *buffer, VkExternalMemoryHandleTypeFlagBitsKHR external_memory_handle_type_flag)
{
	HANDLE handle = {};

	VkMemoryGetWin32HandleInfoKHR handle_info = {};

	handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.pNext      = NULL;
	handle_info.memory     = buffer->GetMemory();
	handle_info.handleType = external_memory_handle_type_flag;

	device->GetMemoryWin32Handle(&handle_info, &handle);

	return handle;
}

Buffer::Buffer(Device *device, Vulkan::Device *vk_device, Vulkan::Buffer *vk_buffer) :
    RHIBuffer(vk_device, vk_buffer->GetDesc())
{
	cudaExternalMemoryHandleDesc cuda_external_memory_handle_desc = {};

#ifdef _WIN64
	cuda_external_memory_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
	cuda_external_memory_handle_desc.handle.win32.handle = GetVkBufferMemHandle(vk_device, vk_buffer, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
	cuda_external_memory_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
	cuda_external_memory_handle_desc.handle.fd = GetVkImageMemHandle(device, texture, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
	cuda_external_memory_handle_desc.size = vk_buffer->GetDesc().size;

	cudaImportExternalMemory(&m_memory, &cuda_external_memory_handle_desc);

	cudaExternalMemoryBufferDesc desc = {};

	desc.size = vk_buffer->GetDesc().size;
	desc.offset = 0;
	desc.flags  = 0;

	cudaExternalMemoryGetMappedBuffer((void **) &m_handle, m_memory, &desc);
}

Buffer::~Buffer()
{
	cudaFree(m_handle);
}

void Buffer::CopyToDevice(const void *data, size_t size, size_t offset)
{
	cudaMemcpy(m_handle, data, size, cudaMemcpyHostToDevice);
}

void Buffer::CopyToHost(void *data, size_t size, size_t offset)
{
	cudaMemcpy(data, m_handle, size, cudaMemcpyDeviceToHost);
}

void *Buffer::Map()
{
	if (m_desc.memory != RHIMemoryUsage::GPU_Only)
	{
		return m_handle;
	}
	return nullptr;
}

void Buffer::Unmap()
{
}

void Buffer::Flush(size_t offset, size_t size)
{
}

void *Buffer::GetHandle() const
{
	return m_handle;
}

uint64_t Buffer::GetDeviceAddress() const
{
	return (uint64_t) m_handle;
}
}        // namespace Ilum::CUDA