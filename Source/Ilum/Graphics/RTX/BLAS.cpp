#include "BLAS.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Command/CommandBuffer.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
void BLAS::add(const VkAccelerationStructureGeometryTrianglesDataKHR &triangle_data)
{
	VkAccelerationStructureGeometryKHR geometry = {};
	geometry.sType                              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType                       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags                              = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.triangles                 = triangle_data;

	m_acceleration_structure_geometries.push_back(geometry);
}

void BLAS::add(const VkAccelerationStructureBuildRangeInfoKHR& range_info)
{
	m_acceleration_structure_build_range_infos.push_back(range_info);
	m_primitive_counts.push_back(range_info.primitiveCount);
}

void BLAS::build(VkBuildAccelerationStructureModeKHR mode)
{
	if (m_primitive_counts.empty())
	{
		return;
	}

	VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info = {};
	build_geometry_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_geometry_info.type                                        = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build_geometry_info.flags                                       = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build_geometry_info.mode                                        = mode;
	if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && m_handle != VK_NULL_HANDLE)
	{
		build_geometry_info.srcAccelerationStructure = m_handle;
		build_geometry_info.dstAccelerationStructure = m_handle;
	}
	build_geometry_info.geometryCount = static_cast<uint32_t>(m_acceleration_structure_geometries.size());
	build_geometry_info.pGeometries   = m_acceleration_structure_geometries.data();

	// Get required build sizes
	VkAccelerationStructureBuildSizesInfoKHR build_sizes_info = {};
	build_sizes_info.sType                                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
	    GraphicsContext::instance()->getLogicalDevice(),
	    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	    &build_geometry_info,
	    m_primitive_counts.data(),
	    &build_sizes_info);

	// Create a buffer for the acceleration structure
	if (!m_buffer || m_buffer.getSize() != build_sizes_info.accelerationStructureSize)
	{
		m_buffer = Buffer(
		    build_sizes_info.accelerationStructureSize,
		    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		    VMA_MEMORY_USAGE_GPU_ONLY);

		VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{};
		acceleration_structure_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		acceleration_structure_create_info.buffer = m_buffer;
		acceleration_structure_create_info.size   = build_sizes_info.accelerationStructureSize;
		acceleration_structure_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(GraphicsContext::instance()->getLogicalDevice(), &acceleration_structure_create_info, nullptr, &m_handle);
	}

	// Get the acceleration structure's handle
	VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info={};
	acceleration_device_address_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	acceleration_device_address_info.accelerationStructure = m_handle;
	m_device_address                                                            = vkGetAccelerationStructureDeviceAddressKHR(GraphicsContext::instance()->getLogicalDevice(), &acceleration_device_address_info);

	// Create a scratch buffer as a temporary storage for the acceleration structure build
	Buffer scratch_buffer(build_sizes_info.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	build_geometry_info.scratchData.deviceAddress = scratch_buffer.getDeviceAddress();
	build_geometry_info.dstAccelerationStructure  = m_handle;

	// Build the acceleration structure on the device via a one-time command buffer submission
	GraphicsContext::instance()->getQueueSystem().waitAll();
	CommandBuffer cmd_buffer(QueueUsage::Compute);
	std::vector<VkAccelerationStructureBuildRangeInfoKHR *> as_build_range_infos;
	for (auto& acceleration_structure_build_range_info : m_acceleration_structure_build_range_infos)
	{
		as_build_range_infos.push_back(&acceleration_structure_build_range_info);
	}
	cmd_buffer.begin();
	vkCmdBuildAccelerationStructuresKHR(
	    cmd_buffer,
	    1,
	    &build_geometry_info,
	    as_build_range_infos.data());
	cmd_buffer.end();
	cmd_buffer.submitIdle();
}
}        // namespace Ilum