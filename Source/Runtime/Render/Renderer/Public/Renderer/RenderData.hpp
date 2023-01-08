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

struct View
{
	struct Info
	{
		glm::mat4 view_matrix;
		glm::mat4 inv_view_matrix;
		glm::mat4 projection_matrix;
		glm::mat4 inv_projection_matrix;
		glm::mat4 view_projection_matrix;
		glm::mat4 inv_view_projection_matrix;
		glm::vec3 position;
		uint32_t  frame_count;
	};

	std::unique_ptr<RHIBuffer> buffer = nullptr;
};

struct GPUScene
{
	struct alignas(16) Instance
	{
		glm::mat4 transform;
		uint32_t  mesh_id      = ~0U;
		uint32_t  material_id  = ~0U;
		uint32_t  animation_id = ~0U;
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

	struct Texture
	{
		std::vector<RHITexture *> texture_2d;
	} textures;

	std::vector<const MaterialData *> materials;

	std::vector<RHISampler *> samplers;

	std::unique_ptr<RHIAccelerationStructure> TLAS = nullptr;
};
}        // namespace Ilum