#pragma once

#include "Graphics/Buffer/Buffer.h"

#include "Scene/Component/Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
struct PerInstanceData
{
	glm::mat4 transform      = {};
	glm::mat4 last_transform = {};

	glm::vec3 bbox_min  = {};
	uint32_t  entity_id = 0;

	glm::vec3 bbox_max    = {};
	uint32_t  material_id = std::numeric_limits<uint32_t>::max();

	alignas(16) uint32_t vertex_offset = 0;
	uint32_t index_offset              = 0;
	uint32_t index_count               = 0;
};

struct PerMeshletData
{
	uint32_t instance_id   = 0;
	uint32_t vertex_offset = 0;
	uint32_t index_offset  = 0;
	uint32_t index_count   = 0;

	glm::vec3 center = {};
	float     radius = 0.f;

	glm::vec3 cone_apex   = {};
	float     cone_cutoff = 0.f;

	alignas(16) glm::vec3 cone_axis = {};
};

struct CullingData
{
	glm::mat4 view;

	glm::mat4 last_view;

	// projection matrix elements
	float P00;
	float P11;
	float znear;
	float zfar;

	float    zbuffer_width;
	float    zbuffer_height;
	uint32_t meshlet_count;
	uint32_t instance_count;
};

struct MeshCountData
{
	uint32_t meshlet_visible_count;
	uint32_t instance_visible_count;
	uint32_t meshlet_invisible_count;
};

struct MaterialData
{
	glm::vec4 base_color = {};

	glm::vec3 emissive_color  = {0.f, 0.f, 0.f};
	float     metallic_factor = 0.f;

	float    roughness_factor   = 0.f;
	float    emissive_intensity = 0.f;
	uint32_t albedo_map         = 0;
	uint32_t normal_map         = 0;

	uint32_t metallic_map  = 0;
	uint32_t roughness_map = 0;
	uint32_t emissive_map  = 0;
	uint32_t ao_map        = 0;

	alignas(16) uint32_t displacement_map = 0;
	float displacement_height             = 0.f;
};

struct CameraData
{
	glm::mat4 view_projection;
	glm::mat4 last_view_projection;
	glm::vec4 frustum[6];
	alignas(16) glm::vec3 position;
};

struct RenderBuffer
{
	// Per instance data buffer
	/*
	struct PerInstanceData
{
	mat4 world_transform;
	mat4 last_world_transform;
	mat4 pre_transform;

	vec3 bbox_min;
	uint entity_id;

	vec3 bbox_max;
	uint material_id;

	uint vertex_offset;
	uint index_offset;
	uint index_count;
};
	*/
	Buffer Instance_Buffer = Buffer(1024 * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Per material buffer
	/*
	struct MaterialData
{
	vec4 base_color;

	vec3 emissive_color;
	float metallic_factor;

	float roughness_factor;
	float emissive_intensity;
	uint albedo_map;
	uint normal_map;

	uint metallic_map;
	uint roughness_map;
	uint emissive_map;
	uint ao_map;

	uint displacement_map;
	float displacement_height;
};
	*/
	Buffer Material_Buffer = Buffer(1024 * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Instance visibility buffer
	Buffer Instance_Visibility_Buffer = Buffer(1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Per meshlet data buffer
	/*
	struct PerMeshletData
{
	// Vertex
	uint instance_id;
	uint vertex_offset;
	uint index_offset;
	uint index_count;

	vec3 center;
	float radius;

	vec3 cone_apex;
	float cone_cutoff;

	vec3 cone_axis;
};

	*/
	Buffer Meshlet_Buffer = Buffer(1024 * sizeof(PerMeshletData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Meshlet - index instance from meshlet
	Buffer Draw_Buffer = Buffer(1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Indirect draw command
	Buffer Command_Buffer = Buffer(1024 * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Draw count buffer:
	/*
	struct CountData
{
	uint actual_draw;
	uint meshlet_visible_count;
    uint instance_visible_count;
    uint meshlet_invisible_count;
    uint instance_invisible_count;
    uint meshlet_total_count;
    uint instance_total_count;
};
*/
	Buffer Count_Buffer = Buffer(sizeof(uint32_t) * 7, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

	// Culling data buffer
	/*
	struct CullingData
{
	mat4 view;

    mat4 last_view;

    float P00;
    float P11;
    float znear;
    float zfar;

    float zbuffer_width;
    float zbuffer_height;
    uint meshlet_count;
    uint instance_count;
};
	*/
	Buffer Culling_Buffer = Buffer(sizeof(CullingData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Camera buffer
	/*
	struct CameraData 
{
    mat4 view_projection;
	mat4 last_view_projection;
	vec4 frustum[6];
	vec3 position;
};
	*/
	Buffer Camera_Buffer = Buffer(sizeof(CameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Static Vertex Buffer for meshlet rendering
	Buffer Static_Vertex_Buffer;

	// Static Index Buffer for meshlet rendering
	Buffer Static_Index_Buffer;

	// Dynamic Vertex Buffer for dynamic mesh rendering
	Buffer Dynamic_Vertex_Buffer;

	// Dynamic Index Buffer for dynamic mesh rendering
	Buffer Dynamic_Index_Buffer;

	// Directional Light Buffer
	Buffer Directional_Light_Buffer = Buffer(sizeof(cmpt::DirectionalLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Point Light Buffer
	Buffer Point_Light_Buffer = Buffer(sizeof(cmpt::PointLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Spot Light Buffer
	Buffer Spot_Light_Buffer = Buffer(sizeof(cmpt::SpotLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	//RTX Instance Buffer
	Buffer RTXInstance_Buffer = Buffer(1024 * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);
};

struct RenderStats
{
	// Light count
	struct LightCount
	{
		uint32_t directional_light_count = 0;
		uint32_t spot_light_count        = 0;
		uint32_t point_light_count       = 0;
	} light_count;

	uint32_t model_count = 0;

	struct StaticMeshCount
	{
		uint32_t instance_count   = 0;
		uint32_t meshlet_count    = 0;
		uint32_t instance_visible = 0;
		uint32_t meshlet_visible  = 0;
		uint32_t triangle_count   = 0;
	} static_mesh_count;

	struct DynamicMeshCount
	{
		uint32_t instance_count = 0;
		uint32_t triangle_count = 0;
	} dynamic_mesh_count;

	struct CurveCount
	{
		uint32_t instance_count = 0;
		uint32_t vertices_count = 0;
	} curve_count;
};
}        // namespace Ilum