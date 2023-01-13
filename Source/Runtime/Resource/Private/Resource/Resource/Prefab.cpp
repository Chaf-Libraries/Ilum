#include "Resource/Prefab.hpp"

namespace Ilum
{
struct Resource<ResourceType::Prefab>::Impl
{
	Node root;
};

Resource<ResourceType::Prefab>::Resource(RHIContext *rhi_context, const std::string &name):
    IResource(rhi_context, name, ResourceType::Prefab)
{
}

Resource<ResourceType::Prefab>::Resource(RHIContext *rhi_context, const std::string &name, Node &&root) :
    IResource(name)
{
	m_impl       = new Impl;
	m_impl->root = std::move(root);

	std::vector<uint32_t> thumbnail_data;
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Prefab), thumbnail_data, m_impl->root);
}

Resource<ResourceType::Prefab>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::Prefab>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Prefab>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Prefab), thumbnail_data, m_impl->root);
}

const Resource<ResourceType::Prefab>::Node &Resource<ResourceType::Prefab>::GetRoot() const
{
	return m_impl->root;
}
}        // namespace Ilum