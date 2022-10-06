#include "Resource/SceneResource.hpp"

#include <Core/Path.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
TResource<ResourceType::Scene>::TResource(size_t uuid) :
    Resource(uuid)
{
}

TResource<ResourceType::Scene>::TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    Resource(uuid, meta, rhi_context)
{
}

void TResource<ResourceType::Scene>::Load(RHIContext *rhi_context, size_t index)
{
	ResourceType type = ResourceType::None;
	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".meta", type, m_uuid, m_meta);

	m_valid = true;
	m_index = index;
}

void TResource<ResourceType::Scene>::Import(RHIContext *rhi_context, const std::string &path)
{
	entt::registry registry;

	{
		std::ifstream is(path, std::ios::binary);
		InputArchive  archive(is);
		entt::snapshot_loader{registry}
		    .entities(archive)
		    .component<ALL_COMPONENTS>(archive);
	}

	m_meta = fmt::format("Name: {}\nEntities: {}",
	                     Path::GetInstance().GetFileName(path, false),
	                     registry.size());

	{
		std::ofstream os("Asset/Meta/" + std::to_string(m_uuid) + ".meta", std::ios::binary);
		OutputArchive archive(os);
		archive(ResourceType::Scene, m_uuid, m_meta);
		entt::snapshot{registry}
		    .entities(archive)
		    .component<ALL_COMPONENTS>(archive);
	}
}

void TResource<ResourceType::Scene>::Load(Scene *scene)
{
	std::ifstream is("Asset/Meta/" + std::to_string(m_uuid) + ".meta", std::ios::binary);
	InputArchive  archive(is);
	archive(ResourceType::Scene, m_uuid, m_meta);
	entt::snapshot_loader{(*scene)()}
	    .entities(archive)
	    .component<ALL_COMPONENTS>(archive);
}
}        // namespace Ilum