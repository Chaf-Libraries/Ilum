#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Device.hpp"

namespace Ilum
{
AccelerationStructure::AccelerationStructure(RHIDevice *device) :
    p_device(device)
{
}

AccelerationStructure::~AccelerationStructure()
{
	if (m_handle)
	{
		vkDeviceWaitIdle(p_device->m_device);
		vkDestroyAccelerationStructureKHR(p_device->m_device, m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}
}

uint64_t AccelerationStructure::GetDeviceAddress() const
{
	return m_device_address;
}

AccelerationStructure::operator VkAccelerationStructureKHR() const
{
	return m_handle;
}

void AccelerationStructure::Build(VkCommandBuffer cmd_buffer, AccelerationStructureDesc desc)
{
	VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info = {};
	build_geometry_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_geometry_info.type                                        = desc.type;
	build_geometry_info.flags                                       = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

	if (m_handle)
	{
		build_geometry_info.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
		build_geometry_info.srcAccelerationStructure = m_handle;
	}
	else
	{
		build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	}

	build_geometry_info.geometryCount = 1;
	build_geometry_info.pGeometries   = &desc.geometry;

	uint32_t max_primitive_count = desc.range_info.primitiveCount;

	// Get required build sizes
	VkAccelerationStructureBuildSizesInfoKHR build_sizes_info = {};
	build_sizes_info.sType                                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
	    p_device->m_device,
	    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	    &build_geometry_info,
	    &max_primitive_count,
	    &build_sizes_info);

	// Create a buffer for the acceleration structure
	if (!m_buffer || m_buffer->GetSize() != build_sizes_info.accelerationStructureSize)
	{
		BufferDesc buffer_desc   = {};
		buffer_desc.size         = build_sizes_info.accelerationStructureSize;
		buffer_desc.buffer_usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
		buffer_desc.memory_usage = VMA_MEMORY_USAGE_GPU_ONLY;
		m_buffer                 = std::make_unique<Buffer>(p_device, buffer_desc);

		if (m_handle)
		{
			vkDeviceWaitIdle(p_device->m_device);
			vkDestroyAccelerationStructureKHR(p_device->m_device, m_handle, nullptr);
		}

		VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info = {};
		acceleration_structure_create_info.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		acceleration_structure_create_info.buffer                               = *m_buffer;
		acceleration_structure_create_info.size                                 = build_sizes_info.accelerationStructureSize;
		acceleration_structure_create_info.type                                 = desc.type;
		vkCreateAccelerationStructureKHR(p_device->m_device, &acceleration_structure_create_info, nullptr, &m_handle);

		build_geometry_info.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
	}

	// Get the acceleration structure's handle
	VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info = {};
	acceleration_device_address_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	acceleration_device_address_info.accelerationStructure                       = m_handle;
	m_device_address                                                             = vkGetAccelerationStructureDeviceAddressKHR(p_device->m_device, &acceleration_device_address_info);

	// Create a scratch buffer as a temporary storage for the acceleration structure build
	BufferDesc scratch_buffer_desc   = {};
	scratch_buffer_desc.size         = build_sizes_info.accelerationStructureSize;
	scratch_buffer_desc.buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	scratch_buffer_desc.memory_usage = VMA_MEMORY_USAGE_GPU_ONLY;
	auto scratch_buffer              = std::make_unique<Buffer>(p_device, scratch_buffer_desc);

	build_geometry_info.scratchData.deviceAddress = scratch_buffer->GetDeviceAddress();
	build_geometry_info.dstAccelerationStructure  = m_handle;

	VkAccelerationStructureBuildRangeInfoKHR *as_build_range_infos = &desc.range_info;

	vkCmdBuildAccelerationStructuresKHR(
	    cmd_buffer,
	    1,
	    &build_geometry_info,
	    &as_build_range_infos);
}

void AccelerationStructure::SetName(const std::string &name)
{
	VkDebugUtilsObjectNameInfoEXT name_info = {};
	name_info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	name_info.pNext                         = nullptr;
	name_info.objectType                    = VK_OBJECT_TYPE_IMAGE_VIEW;
	name_info.objectHandle                  = (uint64_t) m_handle;
	name_info.pObjectName                   = name.c_str();
	vkSetDebugUtilsObjectNameEXT(p_device->m_device, &name_info);

	m_buffer->SetName(name + "_buffer");
}
}        // namespace Ilum