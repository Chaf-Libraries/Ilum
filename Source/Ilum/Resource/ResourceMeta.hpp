#pragma once

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
struct ResourceMeta
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

struct SceneMeta : ResourceMeta<ResourceType::Scene>
{
	std::string name;
};

struct RenderGraphMeta:ResourceMeta<ResourceType::RenderGraph>
{
	std::string name;
};
}        // namespace Ilum