#include "Buffer.hpp"
#include "../Device/Device.hpp"

namespace Ilum::Graphics
{
Buffer::Buffer(const Device &device) :
    m_device(device)
{
}

Buffer::Buffer(const Device &device, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage) :
    m_device(device), m_size(size), m_usage(usage)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size               = size;
	buffer_create_info.usage              = usage;
	buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = memory_usage;

	VmaAllocationInfo allocation_info;
	vmaCreateBuffer(m_device.GetAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
	{
		VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
		buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		buffer_device_address_info.buffer                       = m_handle;
		m_device_address                                        = vkGetBufferDeviceAddressKHR(device, &buffer_device_address_info);
	}
}

Buffer::~Buffer()
{
	Destroy();
}

Buffer::Buffer(Buffer &&other) noexcept :
    m_device(other.m_device),
    m_allocation(other.m_allocation),
    m_handle(other.m_handle),
    m_usage(other.m_usage),
    m_mapping(other.m_mapping),
    m_size(other.m_size)
{
	other.m_allocation = VK_NULL_HANDLE;
	other.m_handle     = VK_NULL_HANDLE;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept
{
	Destroy();

	m_allocation = other.m_allocation;
	m_handle     = other.m_handle;
	m_usage      = other.m_usage;
	m_mapping    = other.m_mapping;
	m_size       = other.m_size;

	other.m_allocation = VK_NULL_HANDLE;
	other.m_handle     = VK_NULL_HANDLE;

	return *this;
}

Buffer::operator const VkBuffer &() const
{
	return m_handle;
}

const VkBuffer &Buffer::GetHandle() const
{
	return m_handle;
}

VkDeviceSize Buffer::GetSize() const
{
	return m_size;
}

uint64_t Buffer::GetDeviceAddress() const
{
	assert(m_usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
	return m_device_address;
}

bool Buffer::IsMapped() const
{
	return m_mapping != nullptr;
}

uint8_t *Buffer::Map()
{
	if (!IsMapped())
	{
		vmaMapMemory(m_device.GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapping));
	}
	return m_mapping;
}

void Buffer::Unmap()
{
	if (!IsMapped())
	{
		return;
	}
	vmaUnmapMemory(m_device.GetAllocator(), m_allocation);
	m_mapping = nullptr;
}

void Buffer::Flush()
{
	this->Flush(m_size, 0);
}

void Buffer::Flush(VkDeviceSize size, VkDeviceSize offset)
{
	vmaFlushAllocation(m_device.GetAllocator(), m_allocation, offset, size);
}

void Buffer::Copy(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset)
{
	ASSERT(size + offset <= m_size);

	if (m_mapping == nullptr)
	{
		Map();
		std::memcpy(m_mapping + offset, data, size);
		Flush(size, offset);
		Unmap();
	}
	else
	{
		std::memcpy(m_mapping + offset, data, size);
	}
}

void Buffer::CopyFlush(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset)
{
	Copy(data, size, offset);
	if (m_mapping)
	{
		Flush(size, offset);
	}
}

void Buffer::Destroy()
{
	if (m_handle)
	{
		if (m_mapping != nullptr)
		{
			Unmap();
		}
		vmaDestroyBuffer(m_device.GetAllocator(), m_handle, m_allocation);
	}
}
}        // namespace Ilum::Graphics