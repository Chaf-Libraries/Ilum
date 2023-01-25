#include "Buffer.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Queue.hpp"
#include "Synchronization.hpp"

#include <dxgi1_2.h>

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

#ifdef CUDA_ENABLE
	// External memory
	VkExternalMemoryBufferCreateInfo external_create_info = {};

	external_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external_create_info.pNext = nullptr;
#	ifdef _WIN64
	external_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#	else
	external_create_info.handleTypes        = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#	endif
	buffer_create_info.pNext = &external_create_info;

	vkCreateBuffer(static_cast<Device *>(p_device)->GetDevice(), &buffer_create_info, nullptr, &m_handle);

	VkMemoryRequirements memory_req = {};
	vkGetBufferMemoryRequirements(static_cast<Device *>(p_device)->GetDevice(), m_handle, &memory_req);

#	ifdef _WIN64
	WindowsSecurityAttributes winSecurityAttributes;

	VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
	vulkanExportMemoryWin32HandleInfoKHR.sType =
	    VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	vulkanExportMemoryWin32HandleInfoKHR.pNext       = NULL;
	vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
	vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
	    DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR) NULL;
#	endif
	VkExportMemoryAllocateInfoKHR export_memory_allocate_info = {};
	export_memory_allocate_info.sType =
	    VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#	ifdef _WIN64
	export_memory_allocate_info.pNext       = &vulkanExportMemoryWin32HandleInfoKHR;
	export_memory_allocate_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#	else
	export_memory_allocate_info.pNext       = NULL;
	export_memory_allocate_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#	endif

	VkMemoryAllocateFlagsInfo allocate_flags_info = {};
	allocate_flags_info.sType                     = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	if (static_cast<Device *>(p_device)->IsFeatureSupport(RHIFeature::BufferDeviceAddress))
	{
		allocate_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	}

	VkMemoryAllocateInfo allocate_info{};
	allocate_info.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate_info.allocationSize = memory_req.size;
	allocate_info.pNext          = &allocate_flags_info;
	allocate_flags_info.pNext    = &export_memory_allocate_info;

	VkPhysicalDeviceMemoryProperties physical_device_properties = {};
	vkGetPhysicalDeviceMemoryProperties(static_cast<Device *>(p_device)->GetPhysicalDevice(), &physical_device_properties);

	VkMemoryPropertyFlags properties = {};
	switch (desc.memory)
	{
		case RHIMemoryUsage::CPU_TO_GPU:
			properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			break;
		case RHIMemoryUsage::GPU_Only:
			properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			break;
		case RHIMemoryUsage::GPU_TO_CPU:
			properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			break;
		default:
			break;
	}

	for (uint32_t i = 0; i < physical_device_properties.memoryTypeCount; i++)
	{
		if ((memory_req.memoryTypeBits & (1 << i)) &&
		    (physical_device_properties.memoryTypes[i].propertyFlags & properties))
		{
			allocate_info.memoryTypeIndex = i;
			break;
		}
	}
	vkAllocateMemory(static_cast<Device *>(p_device)->GetDevice(), &allocate_info, nullptr, &m_memory);
	vkBindBufferMemory(static_cast<Device *>(p_device)->GetDevice(), m_handle, m_memory, 0);
#else
	VmaAllocationCreateInfo allocation_create_info = {};
	allocation_create_info.usage                   = ToVmaMemoryUsage[desc.memory];

	VmaAllocationInfo allocation_info = {};
	vmaCreateBuffer(static_cast<Device *>(p_device)->GetAllocator(), &buffer_create_info, &allocation_create_info, &m_handle, &m_allocation, &allocation_info);
#endif

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
	vkDeviceWaitIdle(static_cast<Device *>(p_device)->GetDevice());

	Unmap();

	if (m_allocation)
	{
		vmaFreeMemory(static_cast<Device *>(p_device)->GetAllocator(), m_allocation);
		m_allocation = VK_NULL_HANDLE;
	}

	if (m_handle)
	{
		vkDestroyBuffer(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
		m_handle = nullptr;
	}

	if (m_memory)
	{
		vkFreeMemory(static_cast<Device *>(p_device)->GetDevice(), m_memory, nullptr);
		m_memory = VK_NULL_HANDLE;
	}
}

void Buffer::CopyToDevice(const void *data, size_t size, size_t offset)
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
		auto    fence = std::make_unique<Fence>(p_device);
		VkQueue queue = VK_NULL_HANDLE;
		vkGetDeviceQueue(static_cast<Device *>(p_device)->GetDevice(), static_cast<Device *>(p_device)->GetQueueFamily(RHIQueueFamily::Transfer), 0, &queue);
		auto staging_buffer = std::make_unique<Buffer>(p_device, BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU, size});

		{
			std::memcpy((uint8_t *) staging_buffer->Map(), data, size);
			auto cmd_buffer = std::make_unique<Command>(p_device, RHIQueueFamily::Transfer);
			cmd_buffer->Init();
			cmd_buffer->Begin();
			cmd_buffer->CopyBufferToBuffer(staging_buffer.get(), this, size, 0, offset);
			cmd_buffer->End();

			auto vk_cmd_buffer = cmd_buffer->GetHandle();

			VkSubmitInfo submit_info         = {};
			submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount   = 1;
			submit_info.pCommandBuffers      = &vk_cmd_buffer;
			submit_info.signalSemaphoreCount = 0;
			submit_info.pSignalSemaphores    = nullptr;
			submit_info.waitSemaphoreCount   = 0;
			submit_info.pWaitSemaphores      = nullptr;
			submit_info.pWaitDstStageMask    = nullptr;

			vkQueueSubmit(queue, 1, &submit_info, fence ? fence->GetHandle() : nullptr);
			fence->Wait();
		}
	}
}

void Buffer::CopyToHost(void *data, size_t size, size_t offset)
{
	if (m_desc.memory == RHIMemoryUsage::GPU_TO_CPU)
	{
		void *mapped = Map();
		std::memcpy(data, (uint8_t *) mapped + offset, size);
		Unmap();
		Flush(offset, size);
	}
	else
	{
		auto    fence = std::make_unique<Fence>(p_device);
		VkQueue queue = VK_NULL_HANDLE;
		vkGetDeviceQueue(static_cast<Device *>(p_device)->GetDevice(), static_cast<Device *>(p_device)->GetQueueFamily(RHIQueueFamily::Transfer), 0, &queue);
		auto staging_buffer = std::make_unique<Buffer>(p_device, BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU, size});

		{
			auto cmd_buffer = std::make_unique<Command>(p_device, RHIQueueFamily::Transfer);
			cmd_buffer->Init();
			cmd_buffer->Begin();
			cmd_buffer->CopyBufferToBuffer(this, staging_buffer.get(), size, 0, offset);
			cmd_buffer->End();

			auto vk_cmd_buffer = cmd_buffer->GetHandle();

			VkSubmitInfo submit_info         = {};
			submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount   = 1;
			submit_info.pCommandBuffers      = &vk_cmd_buffer;
			submit_info.signalSemaphoreCount = 0;
			submit_info.pSignalSemaphores    = nullptr;
			submit_info.waitSemaphoreCount   = 0;
			submit_info.pWaitSemaphores      = nullptr;
			submit_info.pWaitDstStageMask    = nullptr;

			vkQueueSubmit(queue, 1, &submit_info, fence->GetHandle());
			fence->Wait();

			std::memcpy(data, (uint8_t *) staging_buffer->Map(), size);
		}
	}
}

void *Buffer::Map()
{
	if (!m_mapped)
	{
		if (m_allocation)
		{
			vmaMapMemory(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, reinterpret_cast<void **>(&m_mapped));
		}
		else
		{
			vkMapMemory(static_cast<Device *>(p_device)->GetDevice(), m_memory, 0, m_desc.size, 0, &m_mapped);
		}
	}
	return m_mapped;
}

void Buffer::Unmap()
{
	if (m_mapped)
	{
		if (m_allocation)
		{
			vmaUnmapMemory(static_cast<Device *>(p_device)->GetAllocator(), m_allocation);
		}
		else
		{
			vkUnmapMemory(static_cast<Device *>(p_device)->GetDevice(), m_memory);
		}

		m_mapped = nullptr;
	}
}

void Buffer::Flush(size_t offset, size_t size)
{
	if (m_allocation)
	{
		vmaFlushAllocation(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, 0, m_desc.size);
	}
	else
	{
		VkMappedMemoryRange range = {};

		range.memory = m_memory;
		range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		range.offset = 0;
		range.size   = m_desc.size;
		range.pNext  = nullptr;

		vkFlushMappedMemoryRanges(static_cast<Device *>(p_device)->GetDevice(), 1, &range);
	}
}

VkBuffer Buffer::GetHandle() const
{
	return m_handle;
}

VkDeviceMemory Buffer::GetMemory() const
{
	if (m_memory)
	{
		return m_memory;
	}

	VmaAllocationInfo info = {};
	vmaGetAllocationInfo(static_cast<Device *>(p_device)->GetAllocator(), m_allocation, &info);
	return info.deviceMemory;
}

uint64_t Buffer::GetDeviceAddress() const
{
	return m_device_address;
}
}        // namespace Ilum::Vulkan