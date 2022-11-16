#include "AccelerationStructure.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
AccelerationStructure::AccelerationStructure(RHIDevice *device) :
    RHIAccelerationStructure(device)
{
}

AccelerationStructure ::~AccelerationStructure()
{
	if (m_handle)
	{
		p_device->WaitIdle();
		vkDestroyAccelerationStructureKHR(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}
}

void AccelerationStructure::Update(RHICommand *cmd_buffer, const TLASDesc &desc)
{
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	instances.reserve(desc.instances.size());

	for (auto &instance : desc.instances)
	{
		VkAccelerationStructureInstanceKHR vk_instance = {};

		auto transform = glm::mat3x4(glm::transpose(instance.transform));

		std::memcpy(&vk_instance.transform, &transform, sizeof(VkTransformMatrixKHR));
		vk_instance.instanceCustomIndex                    = 0;
		vk_instance.mask                                   = 0xFF;
		vk_instance.instanceShaderBindingTableRecordOffset = instance.material_id;
		vk_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		vk_instance.accelerationStructureReference         = static_cast<AccelerationStructure *>(instance.blas)->GetDeviceAddress();

		instances.emplace_back(std::move(vk_instance));
	}

	if (!m_instance_buffer || m_instance_buffer->GetDesc().size < instances.size() * sizeof(VkAccelerationStructureInstanceKHR))
	{
		m_instance_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc{
		        "TLAS Instance Buffer",
		        RHIBufferUsage::AccelerationStructure | RHIBufferUsage::Transfer,
		        RHIMemoryUsage::CPU_TO_GPU,
		        instances.size() * sizeof(VkAccelerationStructureInstanceKHR)});
	}

	m_instance_buffer->CopyToDevice(instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR), 0);

	VkAccelerationStructureGeometryKHR as_geometry    = {};
	as_geometry.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	as_geometry.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	as_geometry.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
	as_geometry.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	as_geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
	as_geometry.geometry.instances.data.deviceAddress = m_instance_buffer->GetDeviceAddress();

	VkAccelerationStructureBuildRangeInfoKHR range_info = {};
	range_info.primitiveCount                           = static_cast<uint32_t>(desc.instances.size());

	Update(cmd_buffer, as_geometry, range_info, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);

	if (!desc.name.empty())
	{
		VkDebugUtilsObjectNameInfoEXT info = {};
		info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		info.pObjectName                   = desc.name.c_str();
		info.objectHandle                  = (uint64_t) m_handle;
		info.objectType                    = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;
		static_cast<Device *>(p_device)->SetVulkanObjectName(info);
	}
}

void AccelerationStructure::Update(RHICommand *cmd_buffer, const BLASDesc &desc)
{
	VkAccelerationStructureGeometryKHR as_geometry             = {};
	as_geometry.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	as_geometry.flags                                          = VK_GEOMETRY_OPAQUE_BIT_KHR;
	as_geometry.geometryType                                   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	as_geometry.geometry.triangles.sType                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	as_geometry.geometry.triangles.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
	as_geometry.geometry.triangles.vertexData.deviceAddress    = static_cast<Buffer *>(desc.vertex_buffer)->GetDeviceAddress();
	as_geometry.geometry.triangles.maxVertex                   = desc.vertices_count;
	as_geometry.geometry.triangles.vertexStride                = 64;        // 4 x vec4
	as_geometry.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
	as_geometry.geometry.triangles.indexData.deviceAddress     = static_cast<Buffer *>(desc.index_buffer)->GetDeviceAddress();
	as_geometry.geometry.triangles.transformData.deviceAddress = 0;
	as_geometry.geometry.triangles.transformData.hostAddress   = nullptr;

	VkAccelerationStructureBuildRangeInfoKHR range_info = {};

	range_info.primitiveCount  = desc.indices_count / 3;
	range_info.primitiveOffset = desc.indices_offset * sizeof(uint32_t);
	range_info.firstVertex     = desc.vertices_offset;
	range_info.transformOffset = 0;

	Update(cmd_buffer, as_geometry, range_info, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);

	if (!desc.name.empty())
	{
		VkDebugUtilsObjectNameInfoEXT info = {};
		info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		info.pObjectName                   = desc.name.c_str();
		info.objectHandle                  = (uint64_t) m_handle;
		info.objectType                    = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;
		static_cast<Device *>(p_device)->SetVulkanObjectName(info);
	}
}

VkAccelerationStructureKHR AccelerationStructure::GetHandle() const
{
	return m_handle;
}

uint64_t AccelerationStructure::GetDeviceAddress() const
{
	return m_device_address;
}

void AccelerationStructure::Update(RHICommand *cmd_buffer, const VkAccelerationStructureGeometryKHR &geometry, const VkAccelerationStructureBuildRangeInfoKHR &range_info, VkAccelerationStructureTypeKHR type)
{
	VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info = {};

	build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_geometry_info.type  = type;
	build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

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

	// Get required build sizes
	VkAccelerationStructureBuildSizesInfoKHR build_sizes_info = {};
	build_sizes_info.sType                                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
	    static_cast<Device *>(p_device)->GetDevice(),
	    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	    &build_geometry_info,
	    &range_info.primitiveCount,
	    &build_sizes_info);

	// Create a buffer for the acceleration structure
	if (!m_buffer || m_buffer->GetDesc().size != build_sizes_info.accelerationStructureSize)
	{
		m_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc{
		        "AS Buffer",
		        RHIBufferUsage::AccelerationStructure,
		        RHIMemoryUsage::GPU_Only,
		        build_sizes_info.accelerationStructureSize});

		if (m_handle)
		{
			p_device->WaitIdle();
			vkDestroyAccelerationStructureKHR(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
		}

		VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info = {};
		acceleration_structure_create_info.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		acceleration_structure_create_info.buffer                               = m_buffer->GetHandle();
		acceleration_structure_create_info.size                                 = build_sizes_info.accelerationStructureSize;
		acceleration_structure_create_info.type                                 = type;
		vkCreateAccelerationStructureKHR(static_cast<Device *>(p_device)->GetDevice(), &acceleration_structure_create_info, nullptr, &m_handle);

		build_geometry_info.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
	}

	// Get the acceleration structure's handle
	VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info = {};
	acceleration_device_address_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	acceleration_device_address_info.accelerationStructure                       = m_handle;
	m_device_address                                                             = vkGetAccelerationStructureDeviceAddressKHR(static_cast<Device *>(p_device)->GetDevice(), &acceleration_device_address_info);

	if (!m_scratch_buffer || m_scratch_buffer->GetDesc().size < build_sizes_info.buildScratchSize)
	{
		m_scratch_buffer = std::make_unique<Buffer>(
		    p_device,
		    BufferDesc{
		        "AS Scratch Buffer",
		        RHIBufferUsage::UnorderedAccess,
		        RHIMemoryUsage::GPU_Only,
		        build_sizes_info.buildScratchSize});
	}

	build_geometry_info.scratchData.deviceAddress = m_scratch_buffer->GetDeviceAddress();
	build_geometry_info.dstAccelerationStructure  = m_handle;

	VkAccelerationStructureBuildRangeInfoKHR *as_build_range_infos = const_cast<VkAccelerationStructureBuildRangeInfoKHR *>(&range_info);

	// Submit build task
	vkCmdBuildAccelerationStructuresKHR(
	    static_cast<Command *>(cmd_buffer)->GetHandle(),
	    1,
	    &build_geometry_info,
	    &as_build_range_infos);
}
}        // namespace Ilum::Vulkan