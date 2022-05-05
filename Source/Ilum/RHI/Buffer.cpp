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
	vmaCreateBuffer(p_device->GetAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	VkBufferDeviceAddressInfoKHR buffer_device_address_info = {};
	buffer_device_address_info.sType                        = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	buffer_device_address_info.buffer                       = m_handle;

	m_device_address = vkGetBufferDeviceAddress(p_device->GetDevice(), &buffer_device_address_info);
}

Buffer::~Buffer()
{
	if (m_handle && m_allocation)
	{
		vkDeviceWaitIdle(p_device->GetDevice());
		vmaDestroyBuffer(p_device->GetAllocator(), m_handle, m_allocation);
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
		vmaMapMemory(p_device->GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapped_data));
	}
	return m_mapped_data;
}

void Buffer::Unmap()
{
	if (m_mapped_data)
	{
		vmaUnmapMemory(p_device->GetAllocator(), m_allocation);
		m_mapped_data = nullptr;
	}
}
void Buffer::Flush(VkDeviceSize size, VkDeviceSize offset)
{
	vmaFlushAllocation(p_device->GetAllocator(), m_allocation, offset, size);
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
		vkSetDebugUtilsObjectNameEXT(p_device->GetDevice(), &name_info);
	}
}

const BufferDesc &Buffer::GetDesc() const
{
	return m_desc;
}

BufferState::BufferState(VkBufferUsageFlagBits usage)
{
	switch (usage)
	{
		case VK_BUFFER_USAGE_TRANSFER_SRC_BIT:
			access_mask= VK_ACCESS_TRANSFER_READ_BIT;
			break;
		case VK_BUFFER_USAGE_TRANSFER_DST_BIT:
			access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			break;
		case VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		case VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			break;
		case VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			break;
		case VK_BUFFER_USAGE_INDEX_BUFFER_BIT:
			access_mask = VK_ACCESS_INDEX_READ_BIT;
			break;
		case VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
			access_mask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			break;
		case VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:
			access_mask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
			break;
		case VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT:
			access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT:
			access_mask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT:
			access_mask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT:
			access_mask = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR:
			access_mask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
			break;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR:
			access_mask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
			break;
		case VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR:
			access_mask = VK_ACCESS_SHADER_READ_BIT;
			break;
		default:
			access_mask = VK_ACCESS_FLAG_BITS_MAX_ENUM;
			break;
	}

	switch (usage)
	{
		case VK_BUFFER_USAGE_TRANSFER_SRC_BIT:
			stage= VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_BUFFER_USAGE_TRANSFER_DST_BIT:
			stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT:
			stage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case VK_BUFFER_USAGE_INDEX_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:
			stage= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
			break;
		case VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT:
			stage= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			break;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT:
			stage= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT:
			stage= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT:
			stage= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
			break;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR:
			stage= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
			break;
		case VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR:
			stage= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
			break;
		case VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR:
			stage= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			break;
		case VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM:
			stage= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			break;
		default:
			stage= VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
			break;
	}
}
}        // namespace Ilum