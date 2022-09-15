#include "Buffer.hpp"
#include "Command.hpp"
#include "Definitions.hpp"
#include "Device.hpp"
#include "Queue.hpp"
#include "Synchronization.hpp"

namespace Ilum::Vulkan
{
BufferState BufferState::Create(RHIResourceState state)
{
	BufferState vk_state = {};

	switch (state)
	{
		case RHIResourceState::VertexBuffer:
			vk_state.access_mask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case RHIResourceState::IndexBuffer:
			vk_state.access_mask = VK_ACCESS_INDEX_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			break;
		case RHIResourceState::IndirectBuffer:
			vk_state.access_mask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
			break;
		case RHIResourceState::TransferSource:
			vk_state.access_mask = VK_ACCESS_TRANSFER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHIResourceState::TransferDest:
			vk_state.access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case RHIResourceState::AccelerationStructure:
			vk_state.access_mask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
			vk_state.stage       = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
			break;
		case RHIResourceState::ShaderResource:
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
			break;
		case RHIResourceState::UnorderedAccess:
			vk_state.access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vk_state.stage       = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			break;
		case RHIResourceState::ConstantBuffer:
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
	m_desc.size = m_desc.size == 0 ? m_desc.stride * m_desc.count : m_desc.size;

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

	if (!m_desc.name.empty())
	{
		VkDebugUtilsObjectNameInfoEXT info = {};
		info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		info.pObjectName                   = m_desc.name.c_str();
		info.objectHandle                  = (uint64_t) m_handle;
		info.objectType                    = VK_OBJECT_TYPE_BUFFER;
		static_cast<Device *>(p_device)->SetVulkanObjectName(info);
	}
}

Buffer::~Buffer()
{
	Unmap();

	if (m_handle && m_allocation)
	{
		vkDeviceWaitIdle(static_cast<Device *>(p_device)->GetDevice());
		vmaDestroyBuffer(static_cast<Device *>(p_device)->GetAllocator(), m_handle, m_allocation);
	}
}

void Buffer::CopyToDevice(void *data, size_t size, size_t offset)
{
	if (m_desc.memory == RHIMemoryUsage::CPU_TO_GPU)
	{
		void *mapped = Map();
		std::memcpy((uint8_t *) mapped + offset, data, size);
		Unmap();
		Flush(offset, size);
	}
	else
	{
		auto fence          = std::make_unique<Fence>(p_device);
		auto queue          = std::make_unique<Queue>(p_device, RHIQueueFamily::Transfer, 1);
		auto staging_buffer = std::make_unique<Buffer>(p_device, BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU, size});

		{
			std::memcpy((uint8_t *) staging_buffer->Map(), data, size);
			auto cmd_buffer = std::make_unique<Command>(p_device, RHIQueueFamily::Transfer);
			cmd_buffer->Init();
			cmd_buffer->Begin();
			cmd_buffer->CopyBufferToBuffer(staging_buffer.get(), this, size, 0, offset);
			cmd_buffer->End();
			queue->Submit({cmd_buffer.get()});
			queue->Execute(fence.get());
			fence->Wait();
		}
	}
}

void Buffer::CopyToHost(void *data, size_t size, size_t offset)
{
	void *mapped = Map();
	std::memcpy(data, (uint8_t *) mapped + offset, size);
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
		Flush(0, m_desc.size);
	}
}

void Buffer::Flush(size_t offset, size_t size)
{
	vmaFlushAllocation(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, 0, m_desc.size);
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