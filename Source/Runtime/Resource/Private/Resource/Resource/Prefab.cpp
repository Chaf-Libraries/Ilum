#include "Resource/Prefab.hpp"

namespace Ilum
{
struct Resource<ResourceType::Prefab>::Impl
{
	Node root;
};

Resource<ResourceType::Prefab>::Resource(const std::string &name, Node &&root):
    IResource(name)
{
	m_impl       = new Impl;
	m_impl->root = std::move(root);
}

Resource<ResourceType::Prefab>::~Resource()
{
	delete m_impl;
}

const Resource<ResourceType::Prefab>::Node &Resource<ResourceType::Prefab>::GetRoot() const
{
	return m_impl->root;
}
}        // namespace Ilum