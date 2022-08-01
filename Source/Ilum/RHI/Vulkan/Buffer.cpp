#include "Buffer.hpp"
#include "Definitions.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
Buffer::Buffer(RHIDevice *device, const BufferDesc &desc) :
    RHIBuffer(device, desc)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size               = desc.size;
	buffer_create_info.usage              = ToVulkanBufferUsage(desc.usage);
	buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
	
	if (static_cast<Device *>(p_device)->IsBufferDeviceAddressSupport())
	{
		buffer_create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	}

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = ToVmaMemoryUsage[desc.memory];

	VmaAllocationInfo allocation_info = {};
	vmaCreateBuffer(static_cast<Device *>(p_device)->GetAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	if (static_cast<Device *>(p_device)->IsBufferDeviceAddressSupport())
	{
		VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
		buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		buffer_device_address_info.buffer                       = m_handle;
		m_device_address                                        = vkGetBufferDeviceAddress(static_cast<Device *>(p_device)->GetDevice(), &buffer_device_address_info);
	}
}

Buffer::~Buffer()
{
	if (m_handle && m_allocation)
	{
		vkDeviceWaitIdle(static_cast<Device *>(p_device)->GetDevice());
		vmaDestroyBuffer(static_cast<Device *>(p_device)->GetAllocator(), m_handle, m_allocation);
	}
}

void *Buffer::Map()
{
	if (!m_mapped)
	{
		vmaMapMemory(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapped));
	}
	return m_mapped;
}

void Buffer::Unmap()
{
	if (m_mapped)
	{
		vmaUnmapMemory(static_cast<Device *>(p_device)->GetAllocator(), m_allocation);
		m_mapped = nullptr;
		vmaFlushAllocation(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, 0, m_desc.size);
	}
}

VkBuffer Buffer::GetHandle() const
{
	return m_handle;
}

uint64_t Buffer::GetDeviceAddress() const
{
	return m_device_address;
}
}        // namespace Ilum::Vulkan