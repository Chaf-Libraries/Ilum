#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "DescriptorState.hpp"
#include "Device.hpp"
#include "PipelineState.hpp"

#include "Asset/Mesh.hpp"

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

inline int32_t GetLongestCommonPerfix(const std::vector<uint32_t> &morton_code_buffer, uint32_t morton_code_lhs, uint32_t morton_code_rhs)
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

inline void WriteChildren(std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t child, uint32_t parent)
{
	hierarchy[child].parent = parent;
}

inline void WriteParent(std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t parent, uint32_t lchild, uint32_t rchild)
{
	hierarchy[parent].left_child  = lchild;
	hierarchy[parent].right_child = rchild;
}

// Return i, left_child, right_child
inline std::tuple<uint32_t, uint32_t, uint32_t> GenerateHierarchy(const std::vector<uint32_t> &morton_code_buffer, const std::vector<uint32_t> &indices_buffer, std::vector<ShaderInterop::HierarchyNode> &hierarchy, uint32_t left, uint32_t right, uint32_t parent = 0)
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

			return std::make_tuple(i, left_child, right_child);
		}
	}

	return std::make_tuple(~0U, ~0U, ~0U);
}

inline void GenerateAABBs(const std::vector<ShaderInterop::Vertex> &vertices, const std::vector<uint32_t> &indices, const std::vector<ShaderInterop::HierarchyNode> &hierarchy, std::vector<AABB> &aabbs)
{
	// Build from leaves
	for (int32_t node = static_cast<int32_t>(aabbs.size()) - 1; node >= 0; node--)
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
			aabbs[node].Merge(aabbs[left]);
		}

		if (right != ~0U)
		{
			aabbs[node].Merge(aabbs[right]);
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

		Buffer morton_codes_buffer(p_device, BufferDesc(sizeof(uint32_t), primitive_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
		Buffer hierarchy_flag_buffer(p_device, BufferDesc(sizeof(uint32_t), primitive_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));

		if (!m_primitive_indices_buffer || m_primitive_indices_buffer->GetSize() < primitive_count * sizeof(uint32_t))
		{
			m_primitive_indices_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(uint32_t), primitive_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
		}

		if (!m_hierarchy_buffer || m_hierarchy_buffer->GetSize() < (primitive_count * 2 - 1) * sizeof(ShaderInterop::HierarchyNode))
		{
			m_hierarchy_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::HierarchyNode), primitive_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
		}

		if (!m_aabbs_buffer || m_aabbs_buffer->GetSize() < (primitive_count * 2 - 1) * sizeof(ShaderInterop::AABB))
		{
			m_aabbs_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::AABB), primitive_count * 2 - 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
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
			shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
			shader.type        = ShaderType::HLSL;

			PipelineState pso;
			pso.LoadShader(shader);

			cmd_buffer.Bind(pso);
			cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
			                    .Bind(0, 0, &morton_codes_buffer)
			                    .Bind(0, 1, m_primitive_indices_buffer.get())
			                    .Bind(0, 2, &desc.mesh->GetVertexBuffer())
			                    .Bind(0, 3, &desc.mesh->GetIndexBuffer()));
			cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Dispatch((desc.mesh->GetIndicesCount() / 3 + 32 - 1) / 32);
			cmd_buffer.Transition(
			    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
			     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
			    {});
		}

		// Sorting Morton Codes
		// Reference: https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
		{
			uint32_t max_group_size = 1024;
			uint32_t group_size_x   = 1;

			uint32_t n   = desc.mesh->GetIndicesCount() / 3;
			group_size_x = static_cast<uint32_t>(powf(2.f, std::ceilf(std::log2f(static_cast<float>(n)))) / 2.f);
			if (group_size_x > max_group_size)
			{
				group_size_x = max_group_size;
			}

			const uint32_t group_count = (n + group_size_x * 2 - 1) / (group_size_x * 2);

			struct
			{
				uint32_t h    = 0;
				uint32_t size = 0;
			} push_constants;

			push_constants.size = static_cast<uint32_t>(desc.mesh->GetIndicesCount() / 3);

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
				                    .Bind(0, 1, m_primitive_indices_buffer.get()));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_flip = [&](uint32_t h) {
				cmd_buffer.Bind(global_flip_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_primitive_indices_buffer.get()));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto global_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(global_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_primitive_indices_buffer.get()));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			auto local_disperse = [&](uint32_t h) {
				cmd_buffer.Bind(local_disperse_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_primitive_indices_buffer.get()));
				push_constants.h = h;
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch(group_count);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			};

			uint32_t h = group_size_x * 2;

			local_bms(h);

			h <<= 1;

			uint32_t nn = static_cast<uint32_t>(powf(2, std::ceilf(std::log2f(static_cast<float>(n)))));
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
				                    .Bind(0, 1, m_hierarchy_buffer.get()));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count * 2 - 1 + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_hierarchy_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}

			// Split
			{
				PipelineState split_pso;
				split_pso.LoadShader(split_shader);

				cmd_buffer.Bind(split_pso);
				cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
				                    .Bind(0, 0, &morton_codes_buffer)
				                    .Bind(0, 1, m_hierarchy_buffer.get()));
				cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
				cmd_buffer.Dispatch((primitive_count + 1024 - 1) / 1024);
				cmd_buffer.Transition(
				    {BufferTransition{&morton_codes_buffer, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
				     BufferTransition{m_hierarchy_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
				    {});
			}
		}

		// Generate AABB
		{
			struct
			{
				uint32_t leaf_count = 0;
				uint32_t node_count = 0;
			} push_constants;

			push_constants.leaf_count = primitive_count;
			push_constants.node_count = 2 * primitive_count - 1;

			ShaderDesc aabb_shader  = {};
			aabb_shader.filename    = "./Source/Shaders/BVH/GenerateAABB.hlsl";
			aabb_shader.entry_point = "main";
			aabb_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
			aabb_shader.type        = ShaderType::HLSL;

			PipelineState aabb_pso;
			aabb_pso.LoadShader(aabb_shader);

			cmd_buffer.Bind(aabb_pso);
			cmd_buffer.Bind(cmd_buffer.GetDescriptorState()
			                    .Bind(0, 0, m_primitive_indices_buffer.get())
			                    .Bind(0, 1, &desc.mesh->GetVertexBuffer())
			                    .Bind(0, 2, &desc.mesh->GetIndexBuffer())
			                    .Bind(0, 3, m_hierarchy_buffer.get())
			                    .Bind(0, 4, m_aabbs_buffer.get())
			                    .Bind(0, 5, &hierarchy_flag_buffer));
			cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Dispatch((push_constants.node_count + 1024 - 1) / 1024);
			cmd_buffer.Transition(
			    {BufferTransition{m_primitive_indices_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}},
			     BufferTransition{m_hierarchy_buffer.get(), BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}}},
			    {});
		}

		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);

		std::vector<uint32_t>                     morton_codes;
		std::vector<uint32_t>                     indices;
		std::vector<uint32_t>                     hierarchy_flag;
		std::vector<ShaderInterop::HierarchyNode> hierarchy;
		indices.resize(primitive_count);
		morton_codes.resize(primitive_count);
		hierarchy_flag.resize(primitive_count * 2 - 1);
		hierarchy.resize(primitive_count * 2 - 1);
		std::vector<ShaderInterop::AABB> aabbs(primitive_count * 2 - 1);
		std::memcpy(morton_codes.data(), morton_codes_buffer.Map(), morton_codes_buffer.GetSize());
		std::memcpy(indices.data(), m_primitive_indices_buffer->Map(), m_primitive_indices_buffer->GetSize());
		std::memcpy(aabbs.data(), m_aabbs_buffer->Map(), m_aabbs_buffer->GetSize());
		std::memcpy(hierarchy_flag.data(), hierarchy_flag_buffer.Map(), hierarchy_flag_buffer.GetSize());
		std::memcpy(hierarchy.data(), m_hierarchy_buffer->Map(), m_hierarchy_buffer->GetSize());

		morton_codes_buffer.Unmap();
		m_aabbs_buffer->Unmap();
		m_primitive_indices_buffer->Unmap();
		hierarchy_flag_buffer.Unmap();
		m_hierarchy_buffer->Unmap();

#else
		// CPU BVH Construction
		const auto &vertices = desc.mesh->GetVertices();
		const auto &indices  = desc.mesh->GetIndices();

		glm::vec3 aabb_min = desc.mesh->GetAABB().GetMin();
		glm::vec3 aabb_max = desc.mesh->GetAABB().GetMax();

		// Assigned Morton Codes

		std::vector<uint32_t>                     morton_codes_buffer;
		std::vector<uint32_t>                     indices_buffer;
		std::vector<ShaderInterop::HierarchyNode> hierarchy_buffer;
		std::vector<AABB>                         bvhs;
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
		hierarchy_buffer.resize(2 * morton_codes_buffer.size() - 1, ShaderInterop::HierarchyNode{~0U, ~0U, ~0U});
		{
			// left, right, parent
			std::deque<std::tuple<uint32_t, uint32_t, uint32_t>> tasks;
			tasks.push_back(std::make_tuple(0, static_cast<uint32_t>(morton_codes_buffer.size() - 1), 0));
			while (!tasks.empty())
			{
				const auto &[left, right, parent]        = tasks.front();
				const auto &[i, left_child, right_child] = GenerateHierarchy(morton_codes_buffer, indices_buffer, hierarchy_buffer, left, right, parent);

				if (i != ~0U && left_child != ~0U && right_child != ~0U)
				{
					if (i != left)
					{
						tasks.push_back(std::make_tuple(left, i, left_child));
					}
					if (i + 1 != right)
					{
						tasks.push_back(std::make_tuple(i + 1, right, right_child));
					}
				}

				tasks.pop_front();
			}
		}

		// Building BVH
		bvhs.resize(2 * morton_codes_buffer.size() - 1);
		GenerateAABBs(vertices, indices, hierarchy_buffer, bvhs);

		// Upload buffer
		{
			Buffer hierarchy_staging(p_device, BufferDesc(sizeof(ShaderInterop::HierarchyNode), static_cast<uint32_t>(hierarchy_buffer.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
			Buffer aabbs_staging(p_device, BufferDesc(sizeof(ShaderInterop::AABB), static_cast<uint32_t>(bvhs.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
			Buffer primitive_idx_staging(p_device, BufferDesc(sizeof(uint32_t), static_cast<uint32_t>(indices_buffer.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));

			std::memcpy(hierarchy_staging.Map(), hierarchy_buffer.data(), hierarchy_staging.GetSize());
			std::memcpy(primitive_idx_staging.Map(), indices_buffer.data(), primitive_idx_staging.GetSize());
			ShaderInterop::AABB *aabbs = static_cast<ShaderInterop::AABB *>(aabbs_staging.Map());
			for (uint32_t i = 0; i < bvhs.size(); i++)
			{
				aabbs[i].min_val = glm::vec4(bvhs[i].GetMin(), 0.f);
				aabbs[i].max_val = glm::vec4(bvhs[i].GetMax(), 0.f);
			}

			hierarchy_staging.Unmap();
			primitive_idx_staging.Unmap();
			aabbs_staging.Unmap();

			if (!m_hierarchy_buffer || m_hierarchy_buffer->GetSize() < hierarchy_staging.GetSize())
			{
				m_hierarchy_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::HierarchyNode), static_cast<uint32_t>(hierarchy_buffer.size()), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
			}
			if (!m_aabbs_buffer || m_aabbs_buffer->GetSize() < aabbs_staging.GetSize())
			{
				m_aabbs_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(ShaderInterop::AABB), static_cast<uint32_t>(bvhs.size()), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
			}
			if (!m_primitive_indices_buffer || m_primitive_indices_buffer->GetSize() < primitive_idx_staging.GetSize())
			{
				m_primitive_indices_buffer = std::make_unique<Buffer>(p_device, BufferDesc(sizeof(uint32_t), static_cast<uint32_t>(indices_buffer.size()), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY));
			}

			auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_TRANSFER_BIT);
			cmd_buffer.Begin();
			cmd_buffer.CopyBuffer(BufferCopyInfo{&hierarchy_staging}, BufferCopyInfo{m_hierarchy_buffer.get()}, hierarchy_staging.GetSize());
			cmd_buffer.CopyBuffer(BufferCopyInfo{&aabbs_staging}, BufferCopyInfo{m_aabbs_buffer.get()}, aabbs_staging.GetSize());
			cmd_buffer.CopyBuffer(BufferCopyInfo{&primitive_idx_staging}, BufferCopyInfo{m_primitive_indices_buffer.get()}, primitive_idx_staging.GetSize());
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
Buffer &AccelerationStructure::GetHierarchyBuffer()
{
	return *m_hierarchy_buffer;
}

Buffer &AccelerationStructure::GetBoundingVolumeBuffer()
{
	return *m_aabbs_buffer;
}

Buffer &AccelerationStructure::GetPrimitiveIndicesBuffer()
{
	return *m_primitive_indices_buffer;
}
}        // namespace Ilum