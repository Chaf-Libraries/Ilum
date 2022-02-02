#include "Buffer.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"

namespace Ilum::Vulkan
{
Buffer::Buffer(uint64_t                        size,
               VkBufferUsageFlags              buffer_usage,
               VmaMemoryUsage                  memory_usage,
               VmaAllocationCreateFlags        flags,
               const std::vector<QueueFamily> &queue_families)
{
	m_persistent = (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;

	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.usage              = buffer_usage;
	buffer_info.size               = size;
	if (queue_families.size() >= 2)
	{
		std::vector<uint32_t> queue_family_indices;
		for (auto &queue_family : queue_families)
		{
			queue_family_indices.push_back(RenderContext::GetDevice().GetQueueFamily(queue_family));
		}

		buffer_info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family_indices.size());
		buffer_info.pQueueFamilyIndices   = queue_family_indices.data();
	}

	VmaAllocationCreateInfo memory_info{};
	memory_info.flags = flags;
	memory_info.usage = memory_usage;

	VmaAllocationInfo allocation_info{};
	vmaCreateBuffer(RenderContext::GetDevice().GetAllocator(),
	                &buffer_info, &memory_info,
	                &m_handle, &m_allocation,
	                &allocation_info);

	m_memory = allocation_info.deviceMemory;

	if (m_persistent)
	{
		m_mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
	}
}

Buffer ::~Buffer()
{
	if (m_handle != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
	{
		Unmap();
		vmaDestroyBuffer(RenderContext::GetDevice().GetAllocator(), m_handle, m_allocation);
	}
}

void Buffer::Flush() const
{
	vmaFlushAllocation(RenderContext::GetDevice().GetAllocator(), m_allocation, 0, m_size);
}

uint8_t *Buffer::Map()
{
	if (!m_mapped && !m_mapped_data)
	{
		vmaMapMemory(RenderContext::GetDevice().GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapped_data));
		m_mapped = true;
	}

	return m_mapped_data;
}

void Buffer::Unmap()
{
	if (m_mapped)
	{
		vmaUnmapMemory(RenderContext::GetDevice().GetAllocator(), m_allocation);
		m_mapped      = false;
		m_mapped_data = nullptr;
	}
}

Buffer::operator const VkBuffer &() const
{
	return m_handle;
}

const VkBuffer &Buffer::GetHandle() const
{
	return m_handle;
}

uint64_t Buffer::GetSize() const
{
	return m_size;
}

uint64_t Buffer::GetDeviceAddress()
{
	VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
	buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	buffer_device_address_info.buffer                       = m_handle;
	return vkGetBufferDeviceAddressKHR(RenderContext::GetDevice(), &buffer_device_address_info);
}

void Buffer::Update(const uint8_t *data, size_t size, size_t offset)
{
	if (m_persistent)
	{
		std::copy(data, data + size, m_mapped_data + offset);
		Flush();
	}
	else
	{
		Map();
		std::copy(data, data + size, m_mapped_data + offset);
		Flush();
		Unmap();
	}
}

void Buffer::Update(const void *data, size_t size, size_t offset)
{
	Update(reinterpret_cast<const uint8_t *>(data), size, offset);
}

void Buffer::SetName(const std::string& name)
{
	VKDebugger::SetName(m_handle, name.c_str());
}
}        // namespace Ilum::Vulkan