#include "Resource/Prefab.hpp"

namespace Ilum
{
struct Resource<ResourceType::Prefab>::Impl
{
	std::string name;
	Node        root;
};

Resource<ResourceType::Prefab>::Resource(const std::string &name, Node &&root)
{
	m_impl = new Impl;
	m_impl->name = name;
	m_impl->root = std::move(root);
}

Resource<ResourceType::Prefab>::~Resource()
{
	delete m_impl;
}

const std::string &Resource<ResourceType::Prefab>::GetName() const
{
	return m_impl->name;
}

const Resource<ResourceType::Prefab>::Node &Resource<ResourceType::Prefab>::GetRoot() const
{
	return m_impl->root;
}
}        // namespace Ilum