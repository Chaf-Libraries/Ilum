#include "Buffer.hpp"
#include "Definitions.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
BufferState BufferState::Create(RHIBufferState state)
{
	BufferState vk_state = {};

	switch (state)
	{
		case RHIBufferState::Vertex:
			vk_state.access_mask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case RHIBufferState::Index:
			vk_state.access_mask = VK_ACCESS_INDEX_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case RHIBufferState::Indirect:
			vk_state.access_mask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
			break;
		case RHIBufferState::TransferSource:
			vk_state.access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHIBufferState::TransferDest:
			vk_state.access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHIBufferState::AccelerationStructure:
			vk_state.access_mask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
			vk_state.stage       = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
			break;
		case RHIBufferState::ShaderResource:
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case RHIBufferState::UnorderedAccess:
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case RHIBufferState::ConstantBuffer:
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		default:
			vk_state.access_mask = VK_ACCESS_FLAG_BITS_MAX_ENUM;
			vk_state.stage       = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
			break;
	}
	return vk_state;
}

Buffer::Buffer(RHIDevice *device, const BufferDesc &desc) :
    RHIBuffer(device, desc)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size               = desc.size;
	buffer_create_info.usage              = ToVulkanBufferUsage(desc.usage);
	buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
	
	if (static_cast<Device *>(p_device)->IsFeatureSupport(RHIFeature::BufferDeviceAddress))
	{
		buffer_create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	}

	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = ToVmaMemoryUsage[desc.memory];

	VmaAllocationInfo allocation_info = {};
	vmaCreateBuffer(static_cast<Device *>(p_device)->GetAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);

	if (static_cast<Device *>(p_device)->IsFeatureSupport(RHIFeature::BufferDeviceAddress))
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