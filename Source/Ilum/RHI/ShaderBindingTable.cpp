#include "ShaderBindingTable.hpp"
#include "Device.hpp"

namespace Ilum
{
ShaderBindingTable::ShaderBindingTable(RHIDevice *device, uint32_t handle_count) :
    p_device(device), m_handle_count(handle_count)
{
	if (handle_count == 0)
	{
		return;
	}
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_pipeline_properties = {};
	raytracing_pipeline_properties.sType                                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties2                                  = {};
	deviceProperties2.sType                                                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext                                                        = &raytracing_pipeline_properties;
	vkGetPhysicalDeviceProperties2(p_device->m_physical_device, &deviceProperties2);

	uint32_t handle_size_aligned = (raytracing_pipeline_properties.shaderGroupHandleSize + raytracing_pipeline_properties.shaderGroupHandleAlignment - 1) & 
		~(raytracing_pipeline_properties.shaderGroupHandleAlignment - 1);

	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.usage              = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	buffer_info.size               = handle_count * raytracing_pipeline_properties.shaderGroupHandleSize;

	VmaAllocationCreateInfo memory_info{};
	memory_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	memory_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VmaAllocationInfo allocation_info{};
	vmaCreateBuffer(p_device->m_allocator,
	                &buffer_info, &memory_info,
	                &m_buffer, &m_allocation,
	                &allocation_info);

	m_memory = allocation_info.deviceMemory;

	VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
	buffer_device_address_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	buffer_device_address_info.buffer = m_buffer;
	m_handle.deviceAddress            = vkGetBufferDeviceAddress(p_device->m_device, &buffer_device_address_info);
	m_handle.stride                   = handle_size_aligned;
	m_handle.size                     = handle_count * handle_size_aligned;

	m_mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
}

ShaderBindingTable::~ShaderBindingTable()
{
	if (m_buffer&& m_allocation)
	{
		vmaDestroyBuffer(p_device->m_allocator, m_buffer, m_allocation);
	}
}

uint8_t *ShaderBindingTable::GetData()
{
	return m_mapped_data;
}

const VkStridedDeviceAddressRegionKHR *ShaderBindingTable::operator&() const
{
	return &m_handle;
}
}        // namespace Ilum