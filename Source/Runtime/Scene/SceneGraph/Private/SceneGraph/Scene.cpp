#include "Scene.hpp"
#include "Component.hpp"
#include "Node.hpp"

namespace Ilum
{
struct Scene::Impl
{
	Impl() = default;

	~Impl()
	{
		nodes.clear();
		components.clear();
	}

	std::string name;

	std::vector<std::unique_ptr<Node>> nodes;

	std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> components;
};

Scene::Scene(const std::string &name)
{
	m_impl = new Impl;

	m_impl->name = name;
}

Scene::~Scene()
{
	delete m_impl;
	m_impl = nullptr;
}

void Scene::SetName(const std::string &name)
{
	m_impl->name = name;
}

const std::string &Scene::GetName() const
{
	return m_impl->name;
}

const std::vector<std::unique_ptr<Node>> &Scene::GetNodes() const
{
	return m_impl->nodes;
}

const std::vector<Node *> Scene::GetRoots() const
{
	std::vector<Node *> roots;
	roots.reserve(m_impl->nodes.size());
	for (auto &node : m_impl->nodes)
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
	return m_impl->nodes.emplace_back(std::make_unique<Node>(m_impl->nodes.empty() ? 0 : m_impl->nodes.back()->GetID() + 1, *this, name)).get();
}

void Scene::EraseNode(Node *node)
{
	std::function<void(Node *, std::vector<Node *> &)> gather_nodes = [&](Node *node, std::vector<Node *>& nodes) {
		nodes.push_back(node);
		for (auto& child : node->GetChildren())
		{
			gather_nodes(child, nodes);
		}
	};

	std::vector<Node *> remove_nodes;
	gather_nodes(node, remove_nodes);

	for (auto iter = m_impl->nodes.begin(); iter != m_impl->nodes.end();)
	{
		if (std::find(remove_nodes.begin(), remove_nodes.end(), iter->get()) != remove_nodes.end())
		{
			iter=m_impl->nodes.erase(iter);
		}
		else
		{
			iter++;
		}
	}
}

std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> &Scene::GetComponents()
{
	return m_impl->components;
}
}        // namespace Ilum