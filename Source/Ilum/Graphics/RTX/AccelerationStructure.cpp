#include "AccelerationStructure.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR type) :
    m_type(type)
{
	VkAccelerationStructureGeometryKHR geometry = {};
	geometry.sType                              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	if (type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
	{
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	}
	else if (type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
	{
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	}

	build(geometry, VkAccelerationStructureBuildRangeInfoKHR{});
}

AccelerationStructure::~AccelerationStructure()
{
	if (m_handle)
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		vkDestroyAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}
}

AccelerationStructure::AccelerationStructure(AccelerationStructure &&other) :
    m_handle(other.m_handle),
    m_type(other.m_type),
    m_device_address(other.m_device_address),
    m_buffer(std::move(other.m_buffer))
{
	other.m_handle = VK_NULL_HANDLE;
}

AccelerationStructure &AccelerationStructure::operator=(AccelerationStructure &&other)
{
	if (m_handle)
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
		vkDestroyAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}

	m_handle         = other.m_handle;
	m_type           = other.m_type;
	m_device_address = other.m_device_address;
	m_buffer         = std::move(other.m_buffer);

	other.m_handle = VK_NULL_HANDLE;

	return *this;
}

const VkAccelerationStructureKHR AccelerationStructure::operator&() const
{
	return m_handle;
}

const VkAccelerationStructureKHR &AccelerationStructure::getHandle() const
{
	return m_handle;
}

uint64_t AccelerationStructure::getDeviceAddress() const
{
	return m_device_address;
}

const Buffer &AccelerationStructure::getBuffer() const
{
	return m_buffer;
}

bool AccelerationStructure::build(VkAccelerationStructureGeometryKHR &geometry, VkAccelerationStructureBuildRangeInfoKHR &range_info)
{
	bool rebuild = false;

	if (range_info.primitiveCount == 0)
	{
		//return false;
	}

	VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info = {};
	build_geometry_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_geometry_info.type                                        = m_type;
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
	build_geometry_info.pGeometries   = &geometry;

	uint32_t max_primitive_count = range_info.primitiveCount;

	// Get required build sizes
	VkAccelerationStructureBuildSizesInfoKHR build_sizes_info = {};
	build_sizes_info.sType                                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
	    GraphicsContext::instance()->getLogicalDevice(),
	    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	    &build_geometry_info,
	    &max_primitive_count,
	    &build_sizes_info);

	// Create a buffer for the acceleration structure
	if (!m_buffer || m_buffer.getSize() != build_sizes_info.accelerationStructureSize)
	{
		m_buffer = Buffer(
		    build_sizes_info.accelerationStructureSize,
		    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		    VMA_MEMORY_USAGE_GPU_ONLY);

		if (m_handle)
		{
			GraphicsContext::instance()->getQueueSystem().waitAll();
			vkDestroyAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
		}

		VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info = {};
		acceleration_structure_create_info.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		acceleration_structure_create_info.buffer                               = m_buffer;
		acceleration_structure_create_info.size                                 = build_sizes_info.accelerationStructureSize;
		acceleration_structure_create_info.type                                 = m_type;
		vkCreateAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), &acceleration_structure_create_info, nullptr, &m_handle);

		rebuild = true;

		build_geometry_info.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
	}

	// Get the acceleration structure's handle
	VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info = {};
	acceleration_device_address_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	acceleration_device_address_info.accelerationStructure                       = m_handle;
	m_device_address                                                             = vkGetAccelerationStructureDeviceAddressKHR(GraphicsContext::instance()->getLogicalDevice(), &acceleration_device_address_info);

	// Create a scratch buffer as a temporary storage for the acceleration structure build
	Buffer scratch_buffer(build_sizes_info.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	build_geometry_info.scratchData.deviceAddress = scratch_buffer.getDeviceAddress();
	build_geometry_info.dstAccelerationStructure  = m_handle;

	// Build the acceleration structure on the device via a one-time command buffer submission
	GraphicsContext::instance()->getQueueSystem().waitAll();
	CommandBuffer                             cmd_buffer(QueueUsage::Compute);
	VkAccelerationStructureBuildRangeInfoKHR *as_build_range_infos = &range_info;

	cmd_buffer.begin();
	vkCmdBuildAccelerationStructuresKHR(
	    cmd_buffer,
	    1,
	    &build_geometry_info,
	    &as_build_range_infos);
	cmd_buffer.end();
	cmd_buffer.submitIdle();

	return rebuild;
}
}        // namespace Ilum