#pragma once

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>
#include <Graphics/Resource/Buffer.hpp>

#include "Scene/Component/Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
struct PerInstanceData
{
	glm::mat4 world_transform      = {};
	glm::mat4 last_world_transform = {};
	glm::mat4 pre_transform        = {};

	glm::vec3 bbox_min  = {};
	uint32_t  entity_id = 0;

	glm::vec3 bbox_max    = {};
	uint32_t  material_id = std::numeric_limits<uint32_t>::max();
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
	uint32_t frustum_enable;
	uint32_t backface_enable;

	uint32_t occlusion_enable;
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
	Graphics::Buffer Instance_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(PerInstanceData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Per material buffer
	Graphics::Buffer Material_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(MaterialData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Instance visibility buffer
	Graphics::Buffer Instance_Visibility_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Per meshlet data buffer
	Graphics::Buffer Meshlet_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(PerMeshletData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Meshlet - index instance from meshlet
	Graphics::Buffer Draw_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Indirect draw command
	Graphics::Buffer Command_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), 1024 * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Draw count buffer:
	Graphics::Buffer Count_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(uint32_t) * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

	// Culling data buffer
	Graphics::Buffer Culling_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(CullingData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Camera buffer
	Graphics::Buffer Camera_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(CameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Static Vertex Graphics::Buffer for meshlet rendering
	Graphics::Buffer Static_Vertex_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice());

	// Static Index Graphics::Buffer for meshlet rendering
	Graphics::Buffer Static_Index_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice());

	// Dynamic Vertex Graphics::Buffer for dynamic mesh rendering
	Graphics::Buffer Dynamic_Vertex_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice());

	// Dynamic Index Graphics::Buffer for dynamic mesh rendering
	Graphics::Buffer Dynamic_Index_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice());

	// Directional Light Graphics::Buffer
	Graphics::Buffer Directional_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(cmpt::DirectionalLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Point Light Graphics::Buffer
	Graphics::Buffer Point_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(cmpt::PointLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Spot Light Graphics::Buffer
	Graphics::Buffer Spot_Light_Buffer = Graphics::Buffer(Graphics::RenderContext::GetDevice(), sizeof(cmpt::SpotLight) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
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