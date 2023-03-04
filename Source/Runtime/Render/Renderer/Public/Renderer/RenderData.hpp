#pragma once

#include <RHI/RHIContext.hpp>
#include <Resource/Resource/Material.hpp>

namespace Ilum
{
struct DummyTexture
{
	std::unique_ptr<RHITexture> white_opaque      = nullptr;
	std::unique_ptr<RHITexture> black_opaque      = nullptr;
	std::unique_ptr<RHITexture> white_transparent = nullptr;
	std::unique_ptr<RHITexture> black_transparent = nullptr;
};

struct LUT
{
	std::unique_ptr<RHITexture> ggx = nullptr;
};

struct View
{
	struct Info
	{
		glm::vec4 frustum[6];
		glm::mat4 view_matrix;
		glm::mat4 inv_view_matrix;
		glm::mat4 projection_matrix;
		glm::mat4 inv_projection_matrix;
		glm::mat4 view_projection_matrix;
		glm::mat4 inv_view_projection_matrix;
		glm::vec3 position;
		uint32_t  frame_count;
		glm::vec2 viewport;
	};

	std::unique_ptr<RHIBuffer> buffer = nullptr;
};

struct GPUScene
{
	struct Instance
	{
		glm::mat4 transform;
		uint32_t  mesh_id      = ~0U;
		uint32_t  material_id  = ~0U;
		uint32_t  animation_id = ~0U;
		uint32_t  visible      = 0U;
	};

	struct MeshBuffer
	{
		std::vector<RHIBuffer *> vertex_buffers;
		std::vector<RHIBuffer *> index_buffers;
		std::vector<RHIBuffer *> meshlet_data_buffers;
		std::vector<RHIBuffer *> meshlet_buffers;

		std::unique_ptr<RHIBuffer> instances = nullptr;

		uint32_t max_meshlet_count = 0;
		uint32_t instance_count    = 0;
	} mesh_buffer;

	struct SkinnedMeshBuffer
	{
		std::vector<RHIBuffer *> vertex_buffers;
		std::vector<RHIBuffer *> index_buffers;
		std::vector<RHIBuffer *> meshlet_data_buffers;
		std::vector<RHIBuffer *> meshlet_buffers;

		std::unique_ptr<RHIBuffer> instances = nullptr;

		uint32_t max_meshlet_count = 0;
		uint32_t instance_count    = 0;
	} skinned_mesh_buffer;

	struct AnimationBuffer
	{
		struct UpdateInfo
		{
			uint32_t count;
			float    time;
		};

		std::vector<RHITexture *> skinned_matrics;
		std::vector<RHIBuffer *>  bone_matrics;

		std::unique_ptr<RHIBuffer> update_info = nullptr;

		uint32_t max_frame_count = 0;
		uint32_t max_bone_count  = 0;
	} animation_buffer;

	struct LightBuffer
	{
		struct Info
		{
			uint32_t point_light_count = 0;
			uint32_t spot_light_count         = 0;
			uint32_t directional_light_count = 0;
			uint32_t rect_light_count         = 0;
		}info;

		std::unique_ptr<RHIBuffer> point_light_buffer = nullptr;
		std::unique_ptr<RHIBuffer> spot_light_buffer = nullptr;
		std::unique_ptr<RHIBuffer> directional_light_buffer = nullptr;
		std::unique_ptr<RHIBuffer> rect_light_buffer        = nullptr;
		std::unique_ptr<RHIBuffer> light_info_buffer        = nullptr;
	} light;

	struct Texture
	{
		std::vector<RHITexture *> texture_2d;

		RHITexture *texture_cube = nullptr;
	} textures;

	struct
	{
		std::vector<const MaterialData *> data;

		std::unique_ptr<RHIBuffer> material_buffer = nullptr;
		std::unique_ptr<RHIBuffer> material_offset = nullptr;
	} material;

	std::vector<RHISampler *> samplers;

	std::unique_ptr<RHIAccelerationStructure> TLAS = nullptr;
};
}        // namespace Ilum