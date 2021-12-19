#pragma once

#include "Graphics/Buffer/Buffer.h"

#include <glm/glm.hpp>

namespace Ilum
{
//struct PerInstanceData
//{
//	// Transform
//	glm::mat4 world_transform = {};
//	glm::mat4 pre_transform   = {};
//
//	// Material
//	glm::vec4 base_color      = {};
//
//	glm::vec3 emissive_color  = {0.f, 0.f, 0.f};
//	float     metallic_factor = 0.f;
//
//	float    roughness_factor   = 0.f;
//	float    emissive_intensity = 0.f;
//	uint32_t albedo_map         = 0;
//	uint32_t normal_map         = 0;
//
//	uint32_t metallic_map  = 0;
//	uint32_t roughness_map = 0;
//	uint32_t emissive_map  = 0;
//	uint32_t ao_map        = 0;
//
//	glm::vec3 min_            = {};
//	float     displacement_height = 0.f;
//
//	glm::vec3 max_            = {};
//	uint32_t displacement_map             = 0;
//
//	alignas(16) uint32_t entity_id = 0;
//};

//struct PerInstanceData
//{
//	glm::mat4 world_transform = {};
//	glm::mat4 pre_transform   = {};
//
//	glm::vec3 bbox_min  = {};
//	uint32_t  entity_id = 0;
//
//	alignas(16) glm::vec3 bbox_max = {};
//};
//
//struct MaterialData
//{
//	glm::vec4 base_color = {};
//
//	glm::vec3 emissive_color  = {0.f, 0.f, 0.f};
//	float     metallic_factor = 0.f;
//
//	float    roughness_factor   = 0.f;
//	float    emissive_intensity = 0.f;
//	uint32_t albedo_map         = 0;
//	uint32_t normal_map         = 0;
//
//	uint32_t metallic_map  = 0;
//	uint32_t roughness_map = 0;
//	uint32_t emissive_map  = 0;
//	uint32_t ao_map        = 0;
//
//	alignas(16) uint32_t displacement_map = 0;
//
//	 float displacement_height = 0.f;
//};
//
//struct PerMeshletData
//{
//	uint32_t instance_id   = 0;
//	uint32_t vertex_offset = 0;
//	uint32_t index_offset  = 0;
//	uint32_t index_count   = 0;
//
//	glm::vec3 center = {};
//	float     radius = 0.f;
//
//	glm::vec3 cone_apex   = {};
//	float     cone_cutoff = 0.f;
//
//	alignas(16) glm::vec3 cone_axis = {};
//};
//
//struct CullingData
//{
//	glm::mat4 view;
//
//	glm::mat4 last_view;
//
//	// projection matrix elements
//	float P00;
//	float P11;
//	float znear;
//	float zfar;
//
//	float    zbuffer_width;
//	float    zbuffer_height;
//	uint32_t frustum_enable;
//	uint32_t backface_enable;
//
//	uint32_t occlusion_enable;
//	uint32_t meshlet_count;
//	uint32_t instance_count;
//};

// Collecting and sorting geometry render data
struct RenderQueue
{
	// Per instance data buffer
	//Buffer Instance_Buffer = Buffer(1024 * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Per material data buffer
	//Buffer Material_Buffer = Buffer(1024 * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Instance visibility buffer
	//Buffer Instance_Visibility_Buffer = Buffer(1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Per meshlet data buffer
	//Buffer Meshlet_Buffer = Buffer(1024 * sizeof(PerMeshletData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Meshlet - instance indexing
	//Buffer Draw_Buffer = Buffer(1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Indirect draw command
	//Buffer Command_Buffer = Buffer(1024 * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Draw count buffer
	//Buffer Count_Buffer = Buffer(sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

	bool update();
};
}        // namespace Ilum