#pragma once

#include <RHI/RHIAccelerationStructure.hpp>
#include <RHI/RHIBuffer.hpp>
#include <RHI/RHITexture.hpp>

#include <Geometry/Bound/AABB.hpp>
#include <Geometry/Vertex.hpp>

#include <string>

namespace Ilum
{
enum class ResourceType
{
	Unknown,
	Texture,
	Model,
	Scene,
	RenderGraph
};

struct Meshlet
{
	struct Bound
	{
		std::array<float, 3> center;
		float                radius;

		std::array<float, 3> cone_axis;
		float                cone_cutoff;
	} bound;

	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t vertices_offset;        // Global offset
	uint32_t vertices_count;

	alignas(16) uint32_t meshlet_vertices_offset;        // Meshlet offset
	uint32_t meshlet_indices_offset;                     // Meshlet offset
};

struct Submesh
{
	std::string name;

	uint32_t index;

	glm::mat4 pre_transform;

	AABB aabb;

	uint32_t vertices_count;
	uint32_t vertices_offset;
	uint32_t indices_count;
	uint32_t indices_offset;
	uint32_t meshlet_count;
	uint32_t meshlet_offset;
	// TODO: Material
};

template <ResourceType _Ty>
struct [[serialization(false), reflection(false)]] ResourceMeta
{
	std::string uuid;

	inline ResourceType GetType()
	{
		return _Ty;
	}
};

struct [[serialization(false), reflection(false)]] TextureMeta : ResourceMeta<ResourceType::Texture>
{
	TextureDesc desc;

	std::unique_ptr<RHITexture> texture   = nullptr;
	std::unique_ptr<RHITexture> thumbnail = nullptr;

	size_t index = ~0U;        // Index in bindless texture array
};

struct [[serialization(false), reflection(false)]] ModelMeta : ResourceMeta<ResourceType::Model>
{
	std::string name;

	uint32_t vertices_count;
	uint32_t triangle_count;

	std::vector<Submesh> submeshes;

	AABB aabb;

	std::unique_ptr<RHIBuffer>                             vertex_buffer         = nullptr;
	std::unique_ptr<RHIBuffer>                             index_buffer          = nullptr;
	std::unique_ptr<RHIBuffer>                             meshlet_vertex_buffer = nullptr;
	std::unique_ptr<RHIBuffer>                             meshlet_index_buffer  = nullptr;
	std::unique_ptr<RHIBuffer>                             per_meshlet_buffer    = nullptr;
	std::vector<std::unique_ptr<RHIAccelerationStructure>> blas;
};

struct [[serialization(false), reflection(false)]] SceneMeta : ResourceMeta<ResourceType::Scene>
{
	std::string name;
};

struct [[serialization(false), reflection(false)]] RenderGraphMeta : ResourceMeta<ResourceType::RenderGraph>
{
	std::string name;
};
}        // namespace Ilum