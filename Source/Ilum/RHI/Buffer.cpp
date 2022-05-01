#include "Buffer.hpp"
#include "Device.hpp"

namespace Ilum
{
Buffer::Buffer(RHIDevice *device, const BufferDesc &desc) :
    p_device(device), m_desc(desc)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size               = desc.size;
	buffer_create_info.usage              = desc.buffer_usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = desc.memory_usage;

	VmaAllocationInfo allocation_info = {};
	vmaCreateBuffer(p_device->m_allocator, &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
	buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	buffer_device_address_info.buffer                       = m_handle;

	m_device_address = vkGetBufferDeviceAddress(p_device->m_device, &buffer_device_address_info);
}

Buffer::~Buffer()
{
	if (m_handle && m_allocation)
	{
		vkDeviceWaitIdle(p_device->m_device);
		vmaDestroyBuffer(p_device->m_allocator, m_handle, m_allocation);
	}
}

VkDeviceSize Buffer::GetSize() const
{
	return m_desc.size;
}

VkBufferUsageFlags Buffer::GetUsage() const
{
	return m_desc.buffer_usage;
}

uint64_t Buffer::GetDeviceAddress() const
{
	return m_device_address;
}

void *Buffer::Map()
{
	if (!m_mapped_data)
	{
		vmaMapMemory(p_device->m_allocator, m_allocation, reinterpret_cast<void **>(&m_mapped_data));
	}
	return m_mapped_data;
}

void Buffer::Unmap()
{
	if (m_mapped_data)
	{
		vmaUnmapMemory(p_device->m_allocator, m_allocation);
		m_mapped_data = nullptr;
	}
}
void Buffer::Flush(VkDeviceSize size, VkDeviceSize offset)
{
	vmaFlushAllocation(p_device->m_allocator, m_allocation, offset, size);
}

Buffer::operator VkBuffer() const
{
	return m_handle;
}

void Buffer::SetName(const std::string &name)
{
	if (name.empty())
	{
		VkDebugUtilsObjectNameInfoEXT name_info = {};
		name_info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		name_info.pNext                         = nullptr;
		name_info.objectType                    = VK_OBJECT_TYPE_BUFFER;
		name_info.objectHandle                  = (uint64_t) m_handle;
		name_info.pObjectName                   = name.c_str();
		vkSetDebugUtilsObjectNameEXT(p_device->m_device, &name_info);
	}
}
}        // namespace Ilum