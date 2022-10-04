#include "Resource/RenderGraphResource.hpp"

#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
TResource<ResourceType::RenderGraph>::TResource(size_t uuid) :
    Resource(uuid)
{
}

TResource<ResourceType::RenderGraph>::TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    Resource(uuid, meta, rhi_context)
{
	m_valid = true;
}

void TResource<ResourceType::RenderGraph>::Load(RHIContext *rhi_context)
{
	ResourceType type = ResourceType::None;
	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", type, m_uuid, m_meta);
}

void TResource<ResourceType::RenderGraph>::Import(RHIContext *rhi_context, const std::string &path)
{
	std::string     editor_layout;
	RenderGraphDesc desc;
	DESERIALIZE(path, desc, editor_layout);

	m_meta = fmt::format("Passes: {}\nTextures: {}\nBuffers: {}", desc.passes.size(), desc.textures.size(), desc.buffers.size());

	SERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", ResourceType::RenderGraph, m_uuid, m_meta, desc, editor_layout);

	m_valid = true;
}

void TResource<ResourceType::RenderGraph>::Load(RenderGraphDesc &desc, std::string &editor_layout)
{
	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", ResourceType::RenderGraph, m_uuid, m_meta, desc, editor_layout);
}
}        // namespace Ilum