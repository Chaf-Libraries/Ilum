#include "Resource/MaterialResource.hpp"

#include <Core/Path.hpp>
#include <RenderCore/MaterialGraph/MaterialGraph.hpp>

namespace Ilum
{
TResource<ResourceType::Material>::TResource(size_t uuid) :
    Resource(uuid)
{
}

TResource<ResourceType::Material>::TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    Resource(uuid, meta, rhi_context)
{
}

void TResource<ResourceType::Material>::Load(RHIContext *rhi_context, size_t index)
{
	ResourceType type = ResourceType::None;

	MaterialGraphDesc desc;

	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".asset", ResourceType::Material, m_uuid, m_meta, desc, m_editor_state);

	m_material_graph = std::make_unique<MaterialGraph>(rhi_context, desc);

	m_valid = true;
	m_index = index;
}

void TResource<ResourceType::Material>::Import(RHIContext *rhi_context, const std::string &path)
{
	MaterialGraphDesc desc;

	DESERIALIZE(path, desc, m_editor_state);

	m_material_graph = std::make_unique<MaterialGraph>(rhi_context, desc);

	m_meta = fmt::format("Name: {}\nNodes: {}\n", Path::GetInstance().GetFileName(path, false), desc.nodes.size());

	SERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".asset", ResourceType::Material, m_uuid, m_meta, desc, m_editor_state);
}

MaterialGraph *TResource<ResourceType::Material>::Get()
{
	return m_material_graph.get();
}

const std::string &TResource<ResourceType::Material>::GetEditorState() const
{
	return m_editor_state;
}

}        // namespace Ilum