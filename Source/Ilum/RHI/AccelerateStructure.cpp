#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "DescriptorState.hpp"
#include "Device.hpp"
#include "PipelineState.hpp"

#include "Asset/Mesh.hpp"

#include "Scene/Component/Transform.hpp"
#include "Scene/Scene.hpp"

#include <numeric>

#define BIT(x) (1 << (x))

#define GPU_BVH

namespace Ilum
{
// https://www.highperformancegraphics.org/wp-content/uploads/2017/Papers-Session3/HPG207_ExtendedMortonCodes.pdf
inline uint32_t MortonCodeFromUnitCoord(const glm::vec3 &unit_coord)
{
	uint32_t       morton_code = 0;
	const uint32_t num_bits    = 10;
	float          max_coord   = powf(2, num_bits);

	glm::vec3      adjusted_coord   = glm::min(glm::max(unit_coord * max_coord, glm::vec3(0.f)), glm::vec3(max_coord - 1.f));
	const uint32_t num_axis         = 3;
	uint32_t       coords[num_axis] = {static_cast<uint32_t>(adjusted_coord.y), static_cast<uint32_t>(adjusted_coord.x), static_cast<uint32_t>(adjusted_coord.z)};
	for (uint32_t bit_index = 0; bit_index < num_bits; bit_index++)
	{
		for (uint32_t axis = 0; axis < num_axis; axis++)
		{
			uint32_t bit = BIT(bit_index) & coords[axis];
			if (bit)
			{
				morton_code |= BIT(bit_index * num_axis + axis);
			}
		}
	}
	return morton_code;
}

inline int32_t GetLongestCommonPerfix(const std::vector<uint32_t> &morton_code_buffer, int32_t morton_code_lhs, int32_t morton_code_rhs)
{
	if (morton_code_lhs >= morton_code_buffer.size() || morton_code_rhs >= morton_code_buffer.size())
	{
		return -1;
	}

	if (morton_code_buffer[morton_code_lhs] != morton_code_buffer[morton_code_rhs])
	{
		return 31 - ShaderInterop::firstbithigh(morton_code_buffer[morton_code_lhs] ^ morton_code_buffer[morton_code_rhs]);
	}
	else
	{
		return 31 + 31 - ShaderInterop::firstbithigh(morton_code_lhs ^ morton_code_rhs);
	}
}

inline void WriteChildren(std::vector<ShaderInterop::BVHNode> &hierarchy, uint32_t child, uint32_t parent)
{
	hierarchy[child].parent = parent;
}

inline void WriteParent(std::vector<ShaderInterop::BVHNode> &hierarchy, uint32_t parent, uint32_t lchild, uint32_t rchild)
{
	hierarchy[parent].left_child  = lchild;
	hierarchy[parent].right_child = rchild;
}

inline void GenerateHierarchy(const std::vector<uint32_t> &morton_code_buffer, const std::vector<uint32_t> &indices_buffer, std::vector<ShaderInterop::BVHNode> &hierarchy)
{
	for (int32_t idx = 0; idx < static_cast<int32_t>(indices_buffer.size()) - 1; idx++)
	{
		int32_t d          = glm::clamp(GetLongestCommonPerfix(morton_code_buffer, idx, idx + 1) - GetLongestCommonPerfix(morton_code_buffer, idx, idx - 1), -1, 1);
		int32_t min_prefix = GetLongestCommonPerfix(morton_code_buffer, idx, idx - d);
		int32_t l_max      = 2;
		while (GetLongestCommonPerfix(morton_code_buffer, idx, idx + l_max * d) > min_prefix)
		{
			l_max = l_max * 2;
		}
		int32_t l = 0;
		for (int32_t t = l_max / 2; t >= 1; t /= 2)
		{
			if (GetLongestCommonPerfix(morton_code_buffer, idx, idx + (l + t) * d) > min_prefix)
			{
				l = l + t;
			}
		}
		int32_t j           = idx + l * d;
		int32_t node_prefix = GetLongestCommonPerfix(morton_code_buffer, idx, j);
		int32_t s           = 0;
		float   n           = 2.f;
		while (true)
		{
			int32_t t = (int32_t) (ceil((float) l / n));
			if (GetLongestCommonPerfix(morton_code_buffer, idx, idx + (s + t) * d) > node_prefix)
			{
				s = s + t;
			}
			n *= 2.f;
			if (t == 1)
			{
				break;
			}
		}

		int32_t leaf_offset = static_cast<int32_t>(indices_buffer.size()) - 1;
		int32_t gamma       = idx + s * d + glm::min(d, 0);
		int32_t left_child  = gamma;
		int32_t right_child = gamma + 1;

		if (glm::min(idx, j) == gamma)
		{
			left_child += leaf_offset;
		}

		if (glm::max(idx, j) == gamma + 1)
		{
			right_child += leaf_offset;
		}

		WriteParent(hierarchy, idx, static_cast<uint32_t>(left_child), static_cast<uint32_t>(right_child));
		WriteChildren(hierarchy, static_cast<uint32_t>(left_child), idx);
		WriteChildren(hierarchy, static_cast<uint32_t>(right_child), idx);
	}
}

// Return i, left_child, right_child
inline std::tuple<uint32_t, uint32_t, uint32_t> GenerateHierarchy_(const std::vector<uint32_t> &morton_code_buffer, const std::vector<uint32_t> &indices_buffer, std::vector<ShaderInterop::BVHNode> &hierarchy, uint32_t left, uint32_t right, uint32_t parent = 0)
{
	uint32_t node_offset = static_cast<uint32_t>(morton_code_buffer.size() - 1);

	for (uint32_t i = left; i < right; i++)
	{
		if (GetLongestCommonPerfix(morton_code_buffer, i, i + 1) > GetLongestCommonPerfix(morton_code_buffer, i + 1, i + 2) || left + 1 == right || i == right - 1)
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

			return std::make_tuple(i, left_child, right_child);
		}
	}

	return std::make_tuple(~0U, ~0U, ~0U);
}

inline void GenerateAABBs(const std::vector<ShaderInterop::Vertex> &vertices, const std::vector<uint32_t> &indices, std::vector<ShaderInterop::BVHNode> &hierarchy)
{
	std::vector<uint32_t> node_flags(hierarchy.size(), 0);

	uint32_t primitive_count = static_cast<uint32_t>(indices.size()) / 3;

	for (uint32_t i = primitive_count - 1; i < hierarchy.size(); i++)
	{
		uint32_t prim_id = hierarchy[i].prim_id;

		glm::vec3 v1 = vertices[indices[prim_id * 3]].position;
		glm::vec3 v2 = vertices[indices[prim_id * 3 + 1]].position;
		glm::vec3 v3 = vertices[indices[prim_id * 3 + 2]].position;

		hierarchy[i].aabb.min_val = glm::vec4(glm::min(v1, glm::min(v2, v3)), 0.f);
		hierarchy[i].aabb.max_val = glm::vec4(glm::max(v1, glm::max(v2, v3)), 0.f);

		node_flags[i] = 1;
	}

	for (uint32_t i = primitive_count - 1; i < hierarchy.size(); i++)
	{
		uint32_t node = hierarchy[i].parent;

		while (node != ~0U)
		{
			uint32_t prev_flag = node_flags[node];
			node_flags[node]++;

			if (prev_flag != 1)
			{
				break;
			}

			uint32_t left_child  = hierarchy[node].left_child;
			uint32_t right_child = hierarchy[node].right_child;

			ShaderInterop::AABB left_aabb  = hierarchy[left_child].aabb;
			ShaderInterop::AABB right_aabb = hierarchy[right_child].aabb;

			hierarchy[node].aabb.min_val = glm::min(left_aabb.min_val, right_aabb.min_val);
			hierarchy[node].aabb.max_val = glm::max(left_aabb.max_val, right_aabb.max_val);

			node = hierarchy[node].parent;
		}
	}
}

inline void GenerateAABBs(const std::vector<ShaderInterop::Instance> &instances, std::vector<ShaderInterop::BVHNode> &hierarchy)
{
	std::vector<uint32_t> node_flags(hierarchy.size(), 0);

	uint32_t primitive_count = static_cast<uint32_t>(instances.size());

	for (uint32_t i = primitive_count - 1; i < hierarchy.size(); i++)
	{
		uint32_t prim_id = hierarchy[i].prim_id;

		hierarchy[i].aabb.min_val = glm::vec4(instances[prim_id].aabb_min, 1.f);
		hierarchy[i].aabb.max_val = glm::vec4(instances[prim_id].aabb_max, 1.f);

		node_flags[i] = 1;
	}

	for (uint32_t i = primitive_count - 1; i < hierarchy.size(); i++)
	{
		uint32_t node = hierarchy[i].parent;

		while (node != ~0U)
		{
			uint32_t prev_flag = node_flags[node];
			node_flags[node]++;

			if (prev_flag != 1)
			{
				break;
			}

			uint32_t left_child  = hierarchy[node].left_child;
			uint32_t right_child = hierarchy[node].right_child;

			ShaderInterop::AABB left_aabb  = hierarchy[left_child].aabb;
			ShaderInterop::AABB right_aabb = hierarchy[right_child].aabb;

			hierarchy[node].aabb.min_val = glm::min(left_aabb.min_val, right_aabb.min_val);
			hierarchy[node].aabb.max_val = glm::max(left_aabb.max_val, right_aabb.max_val);

			node = hierarchy[node].parent;
		}
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
		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		cmd_buffer.Begin();

		uint32_t primitive_count = desc.mesh->GetIndicesCount() / 3;

		Buffer morton_codes_buffer(p_device, BufferDesc(sizeof(uint32_t), primitive_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		Buffer hierarchy_flag_buffer(p_device, BufferDesc(sizeof(uint32_t), primitive_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		Buffer primitive_indices_buffer(p_device, BufferDesc(sizeof(uint32_t), primitive_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));

		if (!m_bvh_buffer || m_bvh_buffer->GetSize() < (primitive_count * 2 - 1) * sizeof(ShaderInterop::BVHNode))
		{
			m_bvh_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), primitive_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		}

		// Assigned Morton Codes
		{
			struct
			{
				glm::vec3 aabb_min        = {};
				uint32_t  primitive_count = {};
				glm::vec3 aabb_max        = {};
			} push_constants;
			push_constants.aabb_min        = desc.mesh->GetAABB().GetMin();
			push_constants.aabb_max        = desc.mesh->GetAABB().GetMax();
			push_constants.primitive_count = desc.mesh->GetIndicesCount() / 3;

			ShaderDesc shader  = {};
			shader.filename    = "./Source/Shaders/BVH/CalculateMortonCodes.hlsl";
			shader.entry_point = "main";
			shader.macros.push_back("BUILD_BLAS");
			shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shader.type  = ShaderType::HLSL;

			PipelineState pso;
			pso.LoadShader(shader);

			cmd_buffer.Bind(pso);
			cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
			                    .Bind(0, 0, &morton_codes_buffer)
			                    .Bind(0, 1, &primitive_indices_buffer)
			                    .Bind(0, 2, &desc.mesh->GetVertexBuffer())
			                    .Bind(0, 3, &desc.mesh->GetIndexBuffer()));
			cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Dispatch((desc.mesh->GetIndicesCount() / 3 + 32 - 1) / 32);
			cmd_buffer.Transition(
			    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
			     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
			    {});
		}

		// Sorting Morton Codes
		// Reference: https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
		{
			uint32_t max_group_size = 1024;

			uint32_t n            = desc.mesh->GetIndicesCount() / 3;
			uint32_t nn           = static_cast<uint32_t>(powf(2, std::ceilf(std::log2f(static_cast<float>(n)))));
			uint32_t group_size_x = nn / 2;
			if (group_size_x > max_group_size)
			{
				group_size_x = max_group_size;
			}

			const uint32_t group_count = nn / (group_size_x * 2);

			struct
			{
				uint32_t h    = 0;
				uint32_t size = 0;
			} push_constants;

			push_constants.size = primitive_count;

			ShaderDesc local_bms_shader  = {};
			local_bms_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			local_bms_shader.entry_point = "main";
			local_bms_shader.macros.push_back("LOCAL_BMS");
			local_bms_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			local_bms_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			local_bms_shader.type  = ShaderType::HLSL;

			ShaderDesc global_flip_shader  = {};
			global_flip_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			global_flip_shader.entry_point = "main";
			global_flip_shader.macros.push_back("GLOBAL_FLIP");
			global_flip_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			global_flip_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			global_flip_shader.type  = ShaderType::HLSL;

			ShaderDesc global_disperse_shader  = {};
			global_disperse_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			global_disperse_shader.entry_point = "main";
			global_disperse_shader.macros.push_back("GLOBAL_DISPERSE");
			global_disperse_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			global_disperse_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			global_disperse_shader.type  = ShaderType::HLSL;

			ShaderDesc local_disperse_shader  = {};
			local_disperse_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			local_disperse_shader.entry_point = "main";
			local_disperse_shader.macros.push_back("LOCAL_DISPERSE");
			local_disperse_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			local_disperse_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			local_disperse_shader.type  = ShaderType::HLSL;

			PipelineState local_bms_pso;
			local_bms_pso.LoadShader(local_bms_shader);

			PipelineState global_flip_pso;
			global_flip_pso.LoadShader(global_flip_shader);

			PipelineState global_disperse_pso;
			global_disperse_pso.LoadShader(global_disperse_shader);

			PipelineState local_disperse_pso;
			local_disperse_pso.LoadShader(local_disperse_shader);

			auto local_bms = [&](uint32_t h) {
				cmd_buffer.Bind(local_bms_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_flip = [&](uint32_t h) {
				cmd_buffer.Bind(global_flip_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(global_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto local_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(local_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			uint32_t h = group_size_x * 2;

			local_bms(h);

			h <<= 1;

			for (; h <= nn; h <<= 1)
			{
				global_flip(h);
				for (uint32_t hh = h >> 1; hh > 1; hh >>= 1)
				{
					if (hh <= group_size_x * 2)
					{
						local_disperse(hh);
						break;
					}
					else
					{
						global_disperse(hh);
					}
				}
			}
		}

		//	Split Hierarchy
		{
			struct
			{
				uint32_t leaf_count = 0;
			} push_constants;

			push_constants.leaf_count = primitive_count;

			ShaderDesc init_shader  = {};
			init_shader.filename    = "./Source/Shaders/BVH/HierarchySplit.hlsl";
			init_shader.entry_point = "main";
			init_shader.macros.push_back("INITIALIZE");
			init_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			init_shader.type  = ShaderType::HLSL;

			ShaderDesc split_shader  = {};
			split_shader.filename    = "./Source/Shaders/BVH/HierarchySplit.hlsl";
			split_shader.entry_point = "main";
			split_shader.macros.push_back("SPLIT");
			split_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			split_shader.type  = ShaderType::HLSL;

			// Init hierarchy
			{
				PipelineState init_pso;
				init_pso.LoadShader(init_shader);

				cmd_buffer.Bind(init_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 1, m_bvh_buffer.get())
				                    .Bind(0, 2, &primitive_indices_buffer));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count * 2 - 1 + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}

			// Split
			{
				PipelineState split_pso;
				split_pso.LoadShader(split_shader);

				cmd_buffer.Bind(split_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_bvh_buffer.get()));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}
		}

		// Generate AABB
		{
			struct
			{
				uint32_t leaf_count = 0;
			} push_constants;

			push_constants.leaf_count = primitive_count;

			ShaderDesc init_shader  = {};
			init_shader.filename    = "./Source/Shaders/BVH/GenerateAABB.hlsl";
			init_shader.entry_point = "main";
			init_shader.macros.push_back("INITIALIZE");
			init_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			init_shader.type  = ShaderType::HLSL;

			ShaderDesc aabb_shader  = {};
			aabb_shader.filename    = "./Source/Shaders/BVH/GenerateAABB.hlsl";
			aabb_shader.entry_point = "main";
			aabb_shader.macros.push_back("GENERATION");
			aabb_shader.macros.push_back("BUILD_BLAS");
			aabb_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			aabb_shader.type  = ShaderType::HLSL;

			// Init aabb
			{
				PipelineState init_pso;
				init_pso.LoadShader(init_shader);

				cmd_buffer.Bind(init_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, m_bvh_buffer.get())
				                    .Bind(0, 1, &hierarchy_flag_buffer));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count * 2 - 1 + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&hierarchy_flag_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}

			// Generate aabb
			{
				PipelineState aabb_pso;
				aabb_pso.LoadShader(aabb_shader);
				cmd_buffer.Bind(aabb_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, m_bvh_buffer.get())
				                    .Bind(0, 1, &hierarchy_flag_buffer)
				                    .Bind(0, 2, &desc.mesh->GetVertexBuffer())
				                    .Bind(0, 3, &desc.mesh->GetIndexBuffer()));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count + 1024 - 1) / 1024);
			}
		}

		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);
#else
		// CPU BLAS Construction
		const auto &vertices = desc.mesh->GetVertices();
		const auto &indices  = desc.mesh->GetIndices();

		glm::vec3 aabb_min = desc.mesh->GetAABB().GetMin();
		glm::vec3 aabb_max = desc.mesh->GetAABB().GetMax();

		// Assigned Morton Codes
		std::vector<uint32_t>               morton_codes_buffer;
		std::vector<uint32_t>               indices_buffer;
		std::vector<ShaderInterop::BVHNode> bvh_buffer;
		morton_codes_buffer.reserve(indices.size() / 3);
		indices_buffer.resize(indices.size() / 3);
		std::iota(indices_buffer.begin(), indices_buffer.end(), 0);
		for (uint32_t primitive_id = 0; primitive_id < indices.size() / 3; primitive_id++)
		{
			glm::vec3 v1 = vertices[indices[primitive_id * 3]].position;
			glm::vec3 v2 = vertices[indices[primitive_id * 3 + 1]].position;
			glm::vec3 v3 = vertices[indices[primitive_id * 3 + 2]].position;

			glm::vec3 unit_coord = ((v1 + v2 + v3) / 3.f - aabb_min) / glm::max(aabb_max - aabb_min, glm::vec3(0.00001f));

			morton_codes_buffer.push_back(MortonCodeFromUnitCoord(unit_coord));
		}

		// Sorting Primitives
		std::sort(indices_buffer.begin(), indices_buffer.end(), [&](uint32_t lhs, uint32_t rhs) {
			return morton_codes_buffer[lhs] < morton_codes_buffer[rhs];
		});

		std::vector<uint32_t> tmp_morton_codes_buffer(morton_codes_buffer.size());
		for (uint32_t i = 0; i < morton_codes_buffer.size(); i++)
		{
			tmp_morton_codes_buffer[i] = morton_codes_buffer[indices_buffer[i]];
		}
		morton_codes_buffer = std::move(tmp_morton_codes_buffer);

		// Generate Hierarchy
		// LBVH - Lauterbach et al. [2009]
		// [root - node - leaf]
		bvh_buffer.resize(2 * morton_codes_buffer.size() - 1, ShaderInterop::BVHNode{ShaderInterop::AABB{glm::vec4(std::numeric_limits<float>::max()), -glm::vec4(std::numeric_limits<float>::max())}, ~0U, ~0U, ~0U, ~0U});
		GenerateHierarchy(morton_codes_buffer, indices_buffer, bvh_buffer);

		// Building BVH
		for (uint32_t i = 0; i < indices_buffer.size(); i++)
		{
			bvh_buffer[indices_buffer.size() + i - 1].prim_id = indices_buffer[i];
		}

		GenerateAABBs(vertices, indices, bvh_buffer);

		// Upload buffer
		{
			Buffer hierarchy_staging(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), static_cast<uint32_t>(bvh_buffer.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));

			std::memcpy(hierarchy_staging.Map(), bvh_buffer.data(), hierarchy_staging.GetSize());
			hierarchy_staging.Unmap();

			if (!m_bvh_buffer || m_bvh_buffer->GetSize() < hierarchy_staging.GetSize())
			{
				m_bvh_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), static_cast<uint32_t>(bvh_buffer.size()), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
			}

			auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_TRANSFER_BIT);
			cmd_buffer.Begin();
			cmd_buffer.CopyBuffer(BufferCopyInfo{&hierarchy_staging}, BufferCopyInfo{m_bvh_buffer.get()}, hierarchy_staging.GetSize());
			cmd_buffer.End();
			p_device->SubmitIdle(cmd_buffer, VK_QUEUE_TRANSFER_BIT);
		}
#endif
	}
}

void AccelerationStructure::Build(const TLASDesc &desc)
{
	if (p_device->IsRayTracingEnable())
	{
		if (!desc.scene)
		{
			return;
		}

		auto group = desc.scene->GetRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform>);

		if (group.empty())
		{
			return;
		}

		std::vector<VkAccelerationStructureInstanceKHR> acceleration_structure_instances;
		acceleration_structure_instances.reserve(group.size());

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			if (!mesh_renderer.GetMesh())
			{
				return;
			}

			VkAccelerationStructureInstanceKHR acceleration_structure_instance = {};

			auto transform_transpose = glm::mat3x4(glm::transpose(transform.GetWorldTransform()));
			std::memcpy(&acceleration_structure_instance.transform, &transform_transpose, sizeof(VkTransformMatrixKHR));
			acceleration_structure_instance.instanceCustomIndex                    = 0;
			acceleration_structure_instance.mask                                   = 0xFF;
			acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
			acceleration_structure_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			acceleration_structure_instance.accelerationStructureReference         = mesh_renderer.GetMesh()->GetBLAS().GetDeviceAddress();
			acceleration_structure_instances.push_back(acceleration_structure_instance);
		});

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
	else
	{
#ifdef GPU_BVH
		auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		cmd_buffer.Begin();

		AABB scene_aabb;

		auto group = desc.scene->GetRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform>);

		std::vector<Buffer *> instances;
		instances.reserve(group.size());

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			if (mesh_renderer.GetMesh())
			{
				ShaderInterop::Instance *instance = static_cast<ShaderInterop::Instance *>(mesh_renderer.GetBuffer()->Map());
				instances.push_back(mesh_renderer.GetBuffer());

				AABB aabb(instance->aabb_min, instance->aabb_max);
				scene_aabb.Merge(aabb);

				mesh_renderer.GetBuffer()->Unmap();
			}
		});

		uint32_t instance_count = static_cast<uint32_t>(instances.size());

		Buffer morton_codes_buffer(p_device, BufferDesc(sizeof(uint32_t), instance_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		Buffer hierarchy_flag_buffer(p_device, BufferDesc(sizeof(uint32_t), instance_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		Buffer primitive_indices_buffer(p_device, BufferDesc(sizeof(uint32_t), instance_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));

		if (!m_bvh_buffer || m_bvh_buffer->GetSize() < (instance_count * 2 - 1) * sizeof(ShaderInterop::BVHNode))
		{
			m_bvh_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), instance_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
		}

		// Assigned Morton Codes
		{
			struct
			{
				glm::vec3 aabb_min        = {};
				uint32_t  primitive_count = {};
				glm::vec3 aabb_max        = {};
			} push_constants;
			push_constants.aabb_min        = scene_aabb.GetMin();
			push_constants.aabb_max        = scene_aabb.GetMax();
			push_constants.primitive_count = instance_count;

			ShaderDesc shader  = {};
			shader.filename    = "./Source/Shaders/BVH/CalculateMortonCodes.hlsl";
			shader.entry_point = "main";
			shader.macros.push_back("BUILD_TLAS");
			shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shader.type  = ShaderType::HLSL;

			PipelineState pso;
			pso.LoadShader(shader);

			cmd_buffer.Bind(pso);
			cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
			                    .Bind(0, 0, &morton_codes_buffer)
			                    .Bind(0, 1, &primitive_indices_buffer)
			                    .Bind(0, 2, instances));
			cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Dispatch((instance_count + 32 - 1) / 32);
			cmd_buffer.Transition(
			    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
			     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
			    {});
		}

		// Sorting Morton Codes
		// Reference: https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
		{
			uint32_t max_group_size = 1024;

			uint32_t n            = instance_count;
			uint32_t nn           = static_cast<uint32_t>(powf(2, std::ceilf(std::log2f(static_cast<float>(n + 1)))));
			uint32_t group_size_x = nn / 2;
			if (group_size_x > max_group_size)
			{
				group_size_x = max_group_size;
			}

			const uint32_t group_count = nn / (group_size_x * 2);

			struct
			{
				uint32_t h    = 0;
				uint32_t size = 0;
			} push_constants;

			push_constants.size = instance_count;

			ShaderDesc local_bms_shader  = {};
			local_bms_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			local_bms_shader.entry_point = "main";
			local_bms_shader.macros.push_back("LOCAL_BMS");
			local_bms_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			local_bms_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			local_bms_shader.type  = ShaderType::HLSL;

			ShaderDesc global_flip_shader  = {};
			global_flip_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			global_flip_shader.entry_point = "main";
			global_flip_shader.macros.push_back("GLOBAL_FLIP");
			global_flip_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			global_flip_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			global_flip_shader.type  = ShaderType::HLSL;

			ShaderDesc global_disperse_shader  = {};
			global_disperse_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			global_disperse_shader.entry_point = "main";
			global_disperse_shader.macros.push_back("GLOBAL_DISPERSE");
			global_disperse_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			global_disperse_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			global_disperse_shader.type  = ShaderType::HLSL;

			ShaderDesc local_disperse_shader  = {};
			local_disperse_shader.filename    = "./Source/Shaders/BVH/BitonicSort.hlsl";
			local_disperse_shader.entry_point = "main";
			local_disperse_shader.macros.push_back("LOCAL_DISPERSE");
			local_disperse_shader.macros.push_back("GROUP_SIZE=" + std::to_string(group_size_x));
			local_disperse_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			local_disperse_shader.type  = ShaderType::HLSL;

			PipelineState local_bms_pso;
			local_bms_pso.LoadShader(local_bms_shader);

			PipelineState global_flip_pso;
			global_flip_pso.LoadShader(global_flip_shader);

			PipelineState global_disperse_pso;
			global_disperse_pso.LoadShader(global_disperse_shader);

			PipelineState local_disperse_pso;
			local_disperse_pso.LoadShader(local_disperse_shader);

			auto local_bms = [&](uint32_t h) {
				cmd_buffer.Bind(local_bms_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_flip = [&](uint32_t h) {
				cmd_buffer.Bind(global_flip_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(global_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto local_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(local_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, &primitive_indices_buffer));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			uint32_t h = group_size_x * 2;

			local_bms(h);

			h <<= 1;

			for (; h <= nn; h <<= 1)
			{
				global_flip(h);
				for (uint32_t hh = h >> 1; hh > 1; hh >>= 1)
				{
					if (hh <= group_size_x * 2)
					{
						local_disperse(hh);
						break;
					}
					else
					{
						global_disperse(hh);
					}
				}
			}
		}

		//	Split Hierarchy
		{
			struct
			{
				uint32_t leaf_count = 0;
			} push_constants;

			push_constants.leaf_count = instance_count;

			ShaderDesc init_shader  = {};
			init_shader.filename    = "./Source/Shaders/BVH/HierarchySplit.hlsl";
			init_shader.entry_point = "main";
			init_shader.macros.push_back("INITIALIZE");
			init_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			init_shader.type  = ShaderType::HLSL;

			ShaderDesc split_shader  = {};
			split_shader.filename    = "./Source/Shaders/BVH/HierarchySplit.hlsl";
			split_shader.entry_point = "main";
			split_shader.macros.push_back("SPLIT");
			split_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			split_shader.type  = ShaderType::HLSL;

			// Init hierarchy
			{
				PipelineState init_pso;
				init_pso.LoadShader(init_shader);

				cmd_buffer.Bind(init_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 1, m_bvh_buffer.get())
				                    .Bind(0, 2, &primitive_indices_buffer));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((instance_count * 2 - 1 + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}

			// Split
			{
				PipelineState split_pso;
				split_pso.LoadShader(split_shader);

				cmd_buffer.Bind(split_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_bvh_buffer.get()));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((instance_count + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&primitive_indices_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}
		}

		// Generate AABB
		{
			struct
			{
				uint32_t leaf_count = 0;
			} push_constants;

			push_constants.leaf_count = instance_count;

			ShaderDesc init_shader  = {};
			init_shader.filename    = "./Source/Shaders/BVH/GenerateAABB.hlsl";
			init_shader.entry_point = "main";
			init_shader.macros.push_back("INITIALIZE");
			init_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			init_shader.type  = ShaderType::HLSL;

			ShaderDesc aabb_shader  = {};
			aabb_shader.filename    = "./Source/Shaders/BVH/GenerateAABB.hlsl";
			aabb_shader.entry_point = "main";
			aabb_shader.macros.push_back("GENERATION");
			aabb_shader.macros.push_back("BUILD_TLAS");
			aabb_shader.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			aabb_shader.type  = ShaderType::HLSL;

			// Init aabb
			{
				PipelineState init_pso;
				init_pso.LoadShader(init_shader);

				cmd_buffer.Bind(init_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, m_bvh_buffer.get())
				                    .Bind(0, 1, &hierarchy_flag_buffer));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((instance_count * 2 - 1 + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{m_bvh_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{&hierarchy_flag_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}

			// Generate aabb
			{
				PipelineState aabb_pso;
				aabb_pso.LoadShader(aabb_shader);
				cmd_buffer.Bind(aabb_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, m_bvh_buffer.get())
				                    .Bind(0, 1, &hierarchy_flag_buffer)
				                    .Bind(0, 2, instances));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((instance_count + 1024 - 1) / 1024);
			}
		}

		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);
#else
		// CPU TLAS Construction
		auto group = desc.scene->GetRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform>);

		AABB scene_aabb;

		std::vector<ShaderInterop::Instance> instances;
		instances.reserve(group.size());

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			if (mesh_renderer.GetMesh())
			{
				ShaderInterop::Instance *instance = static_cast<ShaderInterop::Instance *>(mesh_renderer.GetBuffer()->Map());
				instances.push_back(*instance);

				AABB aabb(instance->aabb_min, instance->aabb_max);
				scene_aabb.Merge(aabb);

				mesh_renderer.GetBuffer()->Unmap();
			}
		});

		// Assigned Morton Codes
		std::vector<uint32_t>               morton_codes_buffer;
		std::vector<uint32_t>               indices_buffer;
		std::vector<ShaderInterop::BVHNode> bvh_buffer;
		morton_codes_buffer.reserve(instances.size());
		indices_buffer.resize(instances.size());
		std::iota(indices_buffer.begin(), indices_buffer.end(), 0);
		for (auto &instance : instances)
		{
			glm::vec3 unit_coord = ((instance.aabb_min + instance.aabb_max) * 0.5f - scene_aabb.GetMin()) / glm::max(scene_aabb.GetMax() - scene_aabb.GetMin(), glm::vec3(0.00001f));
			morton_codes_buffer.push_back(MortonCodeFromUnitCoord(unit_coord));
		}

		// Sorting Primitives
		std::sort(indices_buffer.begin(), indices_buffer.end(), [&](uint32_t lhs, uint32_t rhs) {
			return morton_codes_buffer[lhs] < morton_codes_buffer[rhs];
		});

		std::vector<uint32_t> tmp_morton_codes_buffer(morton_codes_buffer.size());
		for (uint32_t i = 0; i < morton_codes_buffer.size(); i++)
		{
			tmp_morton_codes_buffer[i] = morton_codes_buffer[indices_buffer[i]];
		}
		morton_codes_buffer = std::move(tmp_morton_codes_buffer);

		// Generate Hierarchy
		// LBVH - Lauterbach et al. [2009]
		// [root - node - leaf]
		bvh_buffer.resize(2 * morton_codes_buffer.size() - 1, ShaderInterop::BVHNode{ShaderInterop::AABB{glm::vec4(std::numeric_limits<float>::max()), -glm::vec4(std::numeric_limits<float>::max())}, ~0U, ~0U, ~0U, ~0U});
		GenerateHierarchy(morton_codes_buffer, indices_buffer, bvh_buffer);

		// Building BVH
		for (uint32_t i = 0; i < indices_buffer.size(); i++)
		{
			bvh_buffer[indices_buffer.size() + i - 1].prim_id = indices_buffer[i];
		}

		GenerateAABBs(instances, bvh_buffer);

		// Upload buffer
		{
			Buffer hierarchy_staging(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), static_cast<uint32_t>(bvh_buffer.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));

			std::memcpy(hierarchy_staging.Map(), bvh_buffer.data(), hierarchy_staging.GetSize());
			hierarchy_staging.Unmap();

			if (!m_bvh_buffer || m_bvh_buffer->GetSize() < hierarchy_staging.GetSize())
			{
				m_bvh_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::BVHNode), static_cast<uint32_t>(bvh_buffer.size()), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
			}

			auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_TRANSFER_BIT);
			cmd_buffer.Begin();
			cmd_buffer.CopyBuffer(BufferCopyInfo{&hierarchy_staging}, BufferCopyInfo{m_bvh_buffer.get()}, hierarchy_staging.GetSize());
			cmd_buffer.End();
			p_device->SubmitIdle(cmd_buffer, VK_QUEUE_TRANSFER_BIT);
		}
#endif
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
Buffer &AccelerationStructure::GetBVHBuffer()
{
	return *m_bvh_buffer;
}
}        // namespace Ilum