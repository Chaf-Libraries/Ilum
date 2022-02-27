#include "ShaderBindingTable.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"

namespace Ilum
{
ShaderBindingTable::ShaderBindingTable(uint32_t handle_count) :
    m_handle_count(handle_count)
{
	if (handle_count == 0)
	{
		return;
	}

	const auto &raytracing_properties = GraphicsContext::instance()->getPhysicalDevice().getRayTracingPipelineProperties();
	uint32_t    handle_size_aligned   = (raytracing_properties.shaderGroupHandleSize + raytracing_properties.shaderGroupHandleAlignment - 1) & ~(raytracing_properties.shaderGroupHandleAlignment - 1);

	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.usage              = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	buffer_info.size               = handle_count * raytracing_properties.shaderGroupHandleSize;

	VmaAllocationCreateInfo memory_info{};
	memory_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	memory_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VmaAllocationInfo allocation_info{};
	vmaCreateBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(),
	                &buffer_info, &memory_info,
	                &m_buffer, &m_allocation,
	                &allocation_info);

	m_memory = allocation_info.deviceMemory;

	VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
	buffer_device_address_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	buffer_device_address_info.buffer = m_buffer;
	m_handle.deviceAddress            = vkGetBufferDeviceAddress(GraphicsContext::instance()->getLogicalDevice(), &buffer_device_address_info);
	m_handle.stride                   = handle_size_aligned;
	m_handle.size                     = handle_count * handle_size_aligned;

	m_mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
}

ShaderBindingTable::~ShaderBindingTable()
{
	if (m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
	{
		vmaDestroyBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_buffer, m_allocation);
	}
}

ShaderBindingTable::ShaderBindingTable(ShaderBindingTable &&other) :
    m_handle(other.m_handle),
    m_buffer(other.m_buffer),
    m_allocation(other.m_allocation),
    m_memory(other.m_memory),
    m_mapped_data(other.m_mapped_data)
{
	other.m_allocation = VK_NULL_HANDLE;
	other.m_buffer     = VK_NULL_HANDLE;
}

ShaderBindingTable &ShaderBindingTable::operator=(ShaderBindingTable &&other)
{
	if (m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE)
	{
		vmaDestroyBuffer(GraphicsContext::instance()->getLogicalDevice().getAllocator(), m_buffer, m_allocation);
	}

	m_handle      = other.m_handle;
	m_buffer      = other.m_buffer;
	m_allocation  = other.m_allocation;
	m_memory      = other.m_memory;
	m_mapped_data = other.m_mapped_data;

	return *this;
}

uint8_t *ShaderBindingTable::getData()
{
	return m_mapped_data;
}

const VkStridedDeviceAddressRegionKHR *ShaderBindingTable::getHandle() const
{
	return &m_handle;
}

const VkStridedDeviceAddressRegionKHR *ShaderBindingTable::operator&() const
{
	return &m_handle;
}
}        // namespace Ilum