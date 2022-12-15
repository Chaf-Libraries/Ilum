#pragma once

#include <RHI/RHIContext.hpp>

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
		glm::vec3 position;
		uint32_t  frame_count;
	}info;

	std::unique_ptr<RHIBuffer> buffer = nullptr;
};

struct GPUScene
{
	std::vector<RHIBuffer *> vertex_buffers;
	std::vector<RHIBuffer *> index_buffers;
	std::vector<RHIBuffer *> meshlet_data_buffers;
	std::vector<RHIBuffer *> meshlet_buffers;

	std::unique_ptr<RHIBuffer> instance_buffer = nullptr;

	std::unique_ptr<RHIAccelerationStructure> TLAS = nullptr;
};
}        // namespace Ilum