//#include "Buffer.h"
//
//#include "Device/LogicalDevice.hpp"
//#include "Graphics/Command/CommandBuffer.hpp"
//#include "Graphics/GraphicsContext.hpp"
//
//namespace Ilum
//{
//Buffer::Buffer(VkDeviceSize size, VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage, const void *data) :
//    m_size(size)
//{
//	auto &allocator = GraphicsContext::instance()->getLogicalDevice().getAllocator();
//
//	VkBufferCreateInfo buffer_create_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
//	buffer_create_info.size        = size;
//	buffer_create_info.usage       = buffer_usage;
//	buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//
//	VmaAllocationCreateInfo allocation_create_info = {};
//	allocation_create_info.usage                   = memory_usage;
//
//	VmaAllocationInfo allocation_info;
//	if (!VK_CHECK(vmaCreateBuffer(allocator, &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info)))
//	{
//		return;
//	}
//
//	if (data)
//	{
//		VkMemoryPropertyFlags properties;
//		vmaGetMemoryTypeProperties(allocator, allocation_info.memoryType, &properties);
//
//		bool is_mappable      = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
//		bool is_host_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
//
//		if (is_mappable)
//		{
//			if (!is_host_coherent)
//			{
//				if (!VK_CHECK(vmaInvalidateAllocation(allocator, m_allocation, 0, size)))
//				{
//					return;
//				}
//			}
//
//			void *mapped = nullptr;
//			if (VK_CHECK(map(&mapped)))
//			{
//				memcpy(mapped, data, size);
//
//				if (!is_host_coherent)
//				{
//					if (!VK_CHECK(vmaFlushAllocation(allocator, m_allocation, 0, size)))
//					{
//						VK_ERROR("Failed to flush allocation!");
//						return;
//					}
//				}
//				unmap();
//			}
//		}
//		else
//		{
//			VK_ERROR("Could not create avaliable buffer. You need to create a CPU side buffer in VMA_MEMORY_USAGE_CPU_ONLY and make a transfer!");
//		}
//	}
//
//	updateDescriptor();
//}
//
//Buffer::~Buffer()
//{
//	auto &logical_device = GraphicsContext::instance()->getLogicalDevice();
//	vmaDestroyBuffer(logical_device.getAllocator(), m_handle, m_allocation);
//}
//
//Buffer::operator const VkBuffer &() const
//{
//	return m_handle;
//}
//
//VkResult Buffer::map(void **data) const
//{
//	return vmaMapMemory(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_allocation, data);
//}
//
//void Buffer::unmap() const
//{
//	vmaUnmapMemory(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_allocation);
//}
//
//const VkBuffer &Buffer::getBuffer() const
//{
//	return m_handle;
//}
//
//const VkDeviceSize Buffer::getSize() const
//{
//	return m_size;
//}
//
//const VkDescriptorBufferInfo &Buffer::getDescriptor() const
//{
//	return m_descriptor;
//}
//
//void Buffer::insertBufferMemoryBarrier(
//    const CommandBuffer &command_buffer,
//    const VkBuffer &     buffer,
//    VkAccessFlags        src_access_mask,
//    VkAccessFlags        dst_access_mask,
//    VkPipelineStageFlags src_stage_mask,
//    VkPipelineStageFlags dst_stage_mask,
//    VkDeviceSize         offset,
//    VkDeviceSize         size)
//{
//	VkBufferMemoryBarrier barrier = {};
//	barrier.sType                 = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
//	barrier.srcAccessMask         = src_access_mask;
//	barrier.dstAccessMask         = dst_access_mask;
//	barrier.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
//	barrier.dstQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
//	barrier.buffer                = buffer;
//	barrier.offset                = offset;
//	barrier.size                  = size;
//	vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 1, &barrier, 0, nullptr);
//}
//
//void Buffer::updateDescriptor()
//{
//	m_descriptor.buffer = m_handle;
//	m_descriptor.offset = 0;
//	m_descriptor.range  = m_size;
//}
//}        // namespace Ilum

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
	vmaCreateBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);
}

Buffer::~Buffer()
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

bool Buffer::isMapped() const
{
	return m_mapping != nullptr;
}

uint8_t *Buffer::map()
{
	if (isMapped())
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
}        // namespace Ilum