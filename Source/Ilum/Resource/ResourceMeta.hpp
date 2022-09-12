#pragma once

#include "Importer/Model/ModelImporter.hpp"

#include <RHI/RHIBuffer.hpp>
#include <RHI/RHITexture.hpp>

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

template <ResourceType _Ty>
struct [[serialization(false), reflection(false)]] ResourceMeta
{
	std::string uuid;

	inline ResourceType GetType()
	{
		return _Ty;
	}
};

struct TextureMeta : ResourceMeta<ResourceType::Texture>
{
	TextureDesc desc;

	std::unique_ptr<RHITexture> texture   = nullptr;
	std::unique_ptr<RHITexture> thumbnail = nullptr;

	size_t index = ~0U;        // Index in bindless texture array
};

struct ModelMeta : ResourceMeta<ResourceType::Model>
{
	std::string name;

	uint32_t vertices_count;
	uint32_t triangle_count;

	std::vector<Submesh> submeshes;
	std::vector<Meshlet> meshlets;

	std::unique_ptr<RHIBuffer> vertex_buffer = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer  = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_vertex_buffer  = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_index_buffer  = nullptr;
	std::unique_ptr<RHIBuffer> per_meshlet_buffer   = nullptr;
};

struct SceneMeta : ResourceMeta<ResourceType::Scene>
{
	std::string name;
};

struct RenderGraphMeta : ResourceMeta<ResourceType::RenderGraph>
{
	std::string name;
};
}        // namespace Ilum