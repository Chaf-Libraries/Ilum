#include "Buffer.h"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
Buffer::Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage) :
    m_size(size)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size               = size;
	buffer_create_info.usage              = usage;
	buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = memory_usage;

	VmaAllocationInfo allocation_info;
	auto result = vmaCreateBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
	{
		VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
		buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		buffer_device_address_info.buffer                       = m_handle;

		m_device_address = vkGetBufferDeviceAddress(GraphicsContext::instance()->getLogicalDevice(), &buffer_device_address_info);
	}
}

Buffer::~Buffer()
{
	destroy();
}

Buffer::Buffer(Buffer &&other) :
    m_allocation(other.m_allocation),
    m_handle(other.m_handle),
    m_mapping(other.m_mapping),
    m_size(other.m_size),
    m_device_address(other.m_device_address)
{
	other.m_allocation = VK_NULL_HANDLE;
	other.m_handle     = VK_NULL_HANDLE;
}

Buffer &Buffer::operator=(Buffer &&other)
{
	destroy();

	m_allocation = other.m_allocation;
	m_handle     = other.m_handle;
	m_mapping    = other.m_mapping;
	m_size       = other.m_size;
	m_device_address = other.m_device_address;

	other.m_allocation = VK_NULL_HANDLE;
	other.m_handle     = VK_NULL_HANDLE;

	return *this;
}

const VkBuffer &Buffer::getBuffer() const
{
	return m_handle;
}

Buffer::operator const VkBuffer &() const
{
	return m_handle;
}

VkDeviceSize Buffer::getSize() const
{
	return m_size;
}

uint64_t Buffer::getDeviceAddress() const
{
	return m_device_address;
}

bool Buffer::isMapped() const
{
	return m_mapping != nullptr;
}

uint8_t *Buffer::map()
{
	if (!isMapped())
	{
		vmaMapMemory(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapping));
	}
	return m_mapping;
}

void Buffer::unmap()
{
	if (!isMapped())
	{
		return;
	}
	vmaUnmapMemory(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_allocation);
	m_mapping = nullptr;
}

void Buffer::flush()
{
	this->flush(m_size, 0);
}

void Buffer::flush(VkDeviceSize size, VkDeviceSize offset)
{
	vmaFlushAllocation(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_allocation, offset, size);
}

void Buffer::copy(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset)
{
	ASSERT(size + offset <= m_size);

	if (m_mapping == nullptr)
	{
		map();
		std::memcpy(m_mapping + offset, data, size);
		flush(size, offset);
		unmap();
	}
	else
	{
		std::memcpy(m_mapping + offset, data, size);
	}
}

void Buffer::copyFlush(const uint8_t *data, VkDeviceSize size, VkDeviceSize offset)
{
	copy(data, size, offset);
	if (m_mapping)
	{
		flush(size, offset);
	}
}

void Buffer::destroy()
{
	if (m_handle)
	{
		if (m_mapping != nullptr)
		{
			unmap();
		}
		vmaDestroyBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_handle, m_allocation);
	}
}
}        // namespace Ilum