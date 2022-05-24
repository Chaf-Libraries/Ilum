#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Device.hpp"

#include "Command.hpp"

#include "Asset/Mesh.hpp"

namespace Ilum
{
inline uint32_t ExpandBits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline uint32_t MortonCodeFromUnitCoord(const glm::vec3 &unit_coord)
{
	float x = std::min(std::max(unit_coord.x * 1024.f, 0.f), 1023.f);
	float y = std::min(std::max(unit_coord.y * 1024.f, 0.f), 1023.f);
	float z = std::min(std::max(unit_coord.z * 1024.f, 0.f), 1023.f);

	uint32_t xx = ExpandBits(asuint(x));
	uint32_t yy = ExpandBits(asuint(y));
	uint32_t zz = ExpandBits(asuint(z));
	return xx * 4 + yy * 2 + zz;
}

inline int32_t GetLongestCommonPerfix(const std::vector<uint32_t> &morton_code_buffer, uint32_t morton_code_lhs, uint32_t morton_code_rhs)
{
	if (morton_code_lhs >= morton_code_buffer.size() || morton_code_rhs >= morton_code_buffer.size())
	{
		return -1;
	}

	if (morton_code_lhs != morton_code_rhs)
	{
		return 31 - ShaderInterop::firstbithigh(morton_code_lhs ^ morton_code_rhs);
	}
	else
	{
		return 31 + 31 - ShaderInterop::firstbithigh(morton_code_lhs ^ morton_code_rhs);
	}
}

inline void WriteChildren(std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t child, uint32_t parent)
{
	hierarchy[child].parent = parent;
}

inline void WriteParent(std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t parent, uint32_t lchild, uint32_t rchild)
{
	hierarchy[parent].left_child  = lchild;
	hierarchy[parent].right_child = rchild;
}

inline void GenerateHierarchy(const std::vector<uint32_t> &morton_code_buffer, const std::vector<uint32_t> &indices_buffer, std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t left, uint32_t right, uint32_t parent = 0, uint32_t depth = 0)
{
	uint32_t node_offset = static_cast<uint32_t>(morton_code_buffer.size() - 1);

	for (uint32_t i = left; i < right; i++)
	{
		if (GetLongestCommonPerfix(morton_code_buffer, i, i + 1) > GetLongestCommonPerfix(morton_code_buffer, i + 1, i + 2) || left + 1 == right)
		{
			uint32_t left_child  = i;
			uint32_t right_child = i + 1;

			// Write left
			if (i == left)
			{
				left_child += node_offset;
			}

			// Write right
			if (i + 1 == right)
			{
				right_child += node_offset;
			}

			WriteChildren(hierarchy, left_child, parent);
			WriteChildren(hierarchy, right_child, parent);
			WriteParent(hierarchy, parent, left_child, right_child);

			// Recursive
			if (i != left)
			{
				GenerateHierarchy(morton_code_buffer, indices_buffer, hierarchy, left, i, left_child, depth + 1);
			}
			if (i + 1 != right)
			{
				GenerateHierarchy(morton_code_buffer, indices_buffer, hierarchy, i + 1, right, right_child, depth + 1);
			}
			break;
		}
	}
}

inline void GenerateAABBs(const std::vector<ShaderInterop::Vertex> &vertices, const std::vector<uint32_t> &indices, const std::vector<ShaderInterop::HierarchyNode> &hierarchy, std::vector<AABB> &aabbs, uint32_t node = 0)
{
	uint32_t primitive_count = static_cast<uint32_t>(indices.size() / 3);

	auto &left  = hierarchy[node].left_child;
	auto &right = hierarchy[node].right_child;

	// Leaf
	if (left == ~0U && right == ~0U)
	{
		uint32_t primitive_id = node - (primitive_count - 1);

		glm::vec3 v1 = vertices[indices[primitive_id * 3]].position;
		glm::vec3 v2 = vertices[indices[primitive_id * 3 + 1]].position;
		glm::vec3 v3 = vertices[indices[primitive_id * 3 + 2]].position;

		aabbs[node].Merge(std::vector{v1, v2, v3});
	}

	// Recursive generation
	if (left != ~0U)
	{
		if (!aabbs[left].IsValid())
		{
			GenerateAABBs(vertices, indices, hierarchy, aabbs, left);
		}
		aabbs[node].Merge(aabbs[left]);
	}

	if (right != ~0U)
	{
		if (!aabbs[right].IsValid())
		{
			GenerateAABBs(vertices, indices, hierarchy, aabbs, right);
		}
		aabbs[node].Merge(aabbs[right]);
	}
}

AccelerationStructure::AccelerationStructure(RHIDevice *device) :
    p_device(device)
{
}

AccelerationStructure::~AccelerationStructure()
{
	if (m_handle)
	{
		vkDeviceWaitIdle(p_device->GetDevice());
		vkDestroyAccelerationStructureKHR(p_device->GetDevice(), m_handle, nullptr);
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
	if (p_device->IsRayTracingEnable())
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
		    p_device->GetDevice(),
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
				vkDeviceWaitIdle(p_device->GetDevice());
				vkDestroyAccelerationStructureKHR(p_device->GetDevice(), m_handle, nullptr);
			}

			VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info = {};
			acceleration_structure_create_info.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			acceleration_structure_create_info.buffer                               = *m_buffer;
			acceleration_structure_create_info.size                                 = build_sizes_info.accelerationStructureSize;
			acceleration_structure_create_info.type                                 = desc.type;
			vkCreateAccelerationStructureKHR(p_device->GetDevice(), &acceleration_structure_create_info, nullptr, &m_handle);

			build_geometry_info.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
			build_geometry_info.srcAccelerationStructure = VK_NULL_HANDLE;
		}

		// Get the acceleration structure's handle
		VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info = {};
		acceleration_device_address_info.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		acceleration_device_address_info.accelerationStructure                       = m_handle;
		m_device_address                                                             = vkGetAccelerationStructureDeviceAddressKHR(p_device->GetDevice(), &acceleration_device_address_info);

		// Create a scratch buffer as a temporary storage for the acceleration structure build
		if (!m_scratch_buffer || m_scratch_buffer->GetSize() != build_sizes_info.accelerationStructureSize)
		{
			BufferDesc scratch_buffer_desc   = {};
			scratch_buffer_desc.size         = build_sizes_info.accelerationStructureSize;
			scratch_buffer_desc.buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
			scratch_buffer_desc.memory_usage = VMA_MEMORY_USAGE_GPU_ONLY;
			m_scratch_buffer                 = std::make_unique<Buffer>(p_device, scratch_buffer_desc);
		}

		build_geometry_info.scratchData.deviceAddress = m_scratch_buffer->GetDeviceAddress();
		build_geometry_info.dstAccelerationStructure  = m_handle;

		VkAccelerationStructureBuildRangeInfoKHR *as_build_range_infos = &desc.range_info;

		vkCmdBuildAccelerationStructuresKHR(
		    cmd_buffer,
		    1,
		    &build_geometry_info,
		    &as_build_range_infos);
	}
}

void AccelerationStructure::Build(const BLASDesc &desc)
{
	if (!desc.mesh)
	{
		return;
	}

	if (p_device->IsRayTracingEnable())
	{
		AccelerationStructureDesc as_desc = {};

		as_desc.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		as_desc.geometry.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		as_desc.geometry.geometryType                                   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		as_desc.geometry.geometry.triangles.sType                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		as_desc.geometry.geometry.triangles.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
		as_desc.geometry.geometry.triangles.vertexData.deviceAddress    = desc.mesh->GetVertexBuffer().GetDeviceAddress();
		as_desc.geometry.geometry.triangles.maxVertex                   = static_cast<uint32_t>(desc.mesh->GetVerticesCount());
		as_desc.geometry.geometry.triangles.vertexStride                = sizeof(ShaderInterop::Vertex);
		as_desc.geometry.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
		as_desc.geometry.geometry.triangles.indexData.deviceAddress     = desc.mesh->GetIndexBuffer().GetDeviceAddress();
		as_desc.geometry.geometry.triangles.transformData.deviceAddress = 0;
		as_desc.geometry.geometry.triangles.transformData.hostAddress   = nullptr;

		as_desc.range_info.primitiveCount  = static_cast<uint32_t>(desc.mesh->GetIndicesCount()) / 3;
		as_desc.range_info.primitiveOffset = 0;
		as_desc.range_info.firstVertex     = 0;
		as_desc.range_info.transformOffset = 0;

		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		cmd_buffer.Begin();
		Build(cmd_buffer, as_desc);
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);
	}
	else
	{
#ifdef GPU_BVH

#else
		// CPU BVH Construction
		const auto &vertices = desc.mesh->GetVertices();
		const auto &indices  = desc.mesh->GetIndices();

		std::map<uint32_t, uint32_t> morton_codes_map;        // morton_code - id

		glm::vec3 aabb_min = desc.mesh->GetAABB().GetMin();
		glm::vec3 aabb_max = desc.mesh->GetAABB().GetMax();

		// Assigned Morton Codes
		for (uint32_t primitive_id = 0; primitive_id < indices.size() / 3; primitive_id++)
		{
			glm::vec3 v1 = vertices[indices[primitive_id * 3]].position;
			glm::vec3 v2 = vertices[indices[primitive_id * 3 + 1]].position;
			glm::vec3 v3 = vertices[indices[primitive_id * 3 + 2]].position;

			glm::vec3 unit_coord = ((v1 + v2 + v3) / 3.f - aabb_min) / glm::max(aabb_max - aabb_min, glm::vec3(0.00001f));

			morton_codes_map[MortonCodeFromUnitCoord(unit_coord)] = primitive_id;
		}

		// Sorting Primitives
		std::vector<uint32_t> morton_codes_buffer;
		std::vector<uint32_t> indices_buffer;
		morton_codes_buffer.reserve(morton_codes_map.size());
		indices_buffer.reserve(morton_codes_map.size());
		for (auto &[morton_code, idx] : morton_codes_map)
		{
			morton_codes_buffer.push_back(morton_code);
			indices_buffer.push_back(idx);
		}

		// Generate Hierarchy
		// LBVH - Lauterbach et al. [2009]
		// [root - node - leaf]
		std::vector<ShaderInterop::HierarchyNode> hierarchy_buffer(2 * morton_codes_buffer.size() - 1, ShaderInterop::HierarchyNode{~0U, ~0U, ~0U});
		GenerateHierarchy(morton_codes_buffer, indices_buffer, hierarchy_buffer, 0, static_cast<uint32_t>(morton_codes_buffer.size() - 1));

		// Building BVH
		std::vector<AABB> bvhs(2 * morton_codes_buffer.size() - 1);
		GenerateAABBs(vertices, indices, hierarchy_buffer, bvhs);
#endif
	}
}

void AccelerationStructure::Build(const TLASDesc &desc)
{
	if (p_device->IsRayTracingEnable())
	{
		if (desc.mesh_instances.empty())
		{
			return;
		}

		std::vector<VkAccelerationStructureInstanceKHR> acceleration_structure_instances;
		acceleration_structure_instances.reserve(desc.mesh_instances.size());

		for (const auto &[transform, mesh] : desc.mesh_instances)
		{
			VkAccelerationStructureInstanceKHR acceleration_structure_instance = {};

			auto transform_transpose = glm::mat3x4(glm::transpose(transform));
			std::memcpy(&acceleration_structure_instance.transform, &transform_transpose, sizeof(VkTransformMatrixKHR));
			acceleration_structure_instance.instanceCustomIndex                    = 0;
			acceleration_structure_instance.mask                                   = 0xFF;
			acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
			acceleration_structure_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			acceleration_structure_instance.accelerationStructureReference         = mesh->GetBLAS().GetDeviceAddress();
			acceleration_structure_instances.push_back(acceleration_structure_instance);
		}

		if (!m_instance_buffer || m_instance_buffer->GetSize() < sizeof(VkAccelerationStructureInstanceKHR) * acceleration_structure_instances.size())
		{
			m_instance_buffer = std::make_unique<Buffer>(
			    p_device,
			    BufferDesc(
			        sizeof(VkAccelerationStructureInstanceKHR),
			        static_cast<uint32_t>(acceleration_structure_instances.size()),
			        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
			            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			        VMA_MEMORY_USAGE_CPU_TO_GPU));
		}
		std::memcpy(m_instance_buffer->Map(), acceleration_structure_instances.data(), m_instance_buffer->GetSize());
		m_instance_buffer->Flush(m_instance_buffer->GetSize());
		m_instance_buffer->Unmap();

		AccelerationStructureDesc as_desc                      = {};
		as_desc.type                                           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		as_desc.geometry.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		as_desc.geometry.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		as_desc.geometry.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
		as_desc.geometry.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		as_desc.geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
		as_desc.geometry.geometry.instances.data.deviceAddress = m_instance_buffer->GetDeviceAddress();
		as_desc.range_info.primitiveCount                      = static_cast<uint32_t>(acceleration_structure_instances.size());
		as_desc.range_info.primitiveOffset                     = 0;
		as_desc.range_info.firstVertex                         = 0;
		as_desc.range_info.transformOffset                     = 0;

		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		cmd_buffer.Begin();
		Build(cmd_buffer, as_desc);
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);
	}
}

void AccelerationStructure::SetName(const std::string &name)
{
	VkDebugUtilsObjectNameInfoEXT name_info = {};
	name_info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	name_info.pNext                         = nullptr;
	name_info.objectType                    = VK_OBJECT_TYPE_IMAGE_VIEW;
	name_info.objectHandle                  = (uint64_t) m_handle;
	name_info.pObjectName                   = name.c_str();
	vkSetDebugUtilsObjectNameEXT(p_device->GetDevice(), &name_info);

	m_buffer->SetName(name + "_buffer");
}
}        // namespace Ilum