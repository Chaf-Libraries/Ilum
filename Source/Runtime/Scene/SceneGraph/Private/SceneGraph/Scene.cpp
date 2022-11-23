#include "Scene.hpp"
#include "Component.hpp"
#include "Node.hpp"

namespace Ilum
{
Scene::Scene(const std::string &name) :
    m_name(m_name)
{
}

Scene::~Scene()
{
	m_nodes.clear();
	m_components.clear();
}

void Scene::SetName(const std::string &name)
{
	m_name = name;
}

const std::string &Scene::GetName() const
{
	return m_name;
}

const std::vector<std::unique_ptr<Node>> &Scene::GetNodes() const
{
	return m_nodes;
}

const std::vector<Node *> Scene::GetRoots() const
{
	std::vector<Node *> roots;
	roots.reserve(m_nodes.size());
	for (auto& node : m_nodes)
	{
		if (node->GetParent() == nullptr)
		{
			roots.emplace_back(node.get());
		}
	}
	return roots;
}

Node *Scene::CreateNode(const std::string &name)
{
	return m_nodes.emplace_back(std::make_unique<Node>(m_nodes.empty() ? 0 : m_nodes.back()->GetID() + 1, *this, name)).get();
}

void Scene::EraseNode(Node *node)
{
	for (auto iter = m_nodes.begin(); iter != m_nodes.end(); iter++)
	{
		if (iter->get() == node)
		{
			m_nodes.erase(iter);
			return;
		}
	}
}
}        // namespace Ilum