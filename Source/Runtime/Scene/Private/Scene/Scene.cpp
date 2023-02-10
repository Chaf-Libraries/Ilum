#include "Scene.hpp"
#include "Component.hpp"
#include "Components/AllComponents.hpp"
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

	bool update = false;
};

Scene::Scene(const std::string &name)
{
	m_impl = new Impl;

	m_impl->name = name;
}

Scene::~Scene()
{
	auto roots = GetRoots();
	for (auto &root : roots)
	{
		EraseNode(root);
	}

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
	std::function<void(Node *, std::vector<Node *> &)> gather_nodes = [&](Node *node, std::vector<Node *> &nodes) {
		nodes.push_back(node);
		for (auto &child : node->GetChildren())
		{
			gather_nodes(child, nodes);
		}
	};

	std::vector<Node *> remove_nodes;
	gather_nodes(node, remove_nodes);
	std::reverse(remove_nodes.begin(), remove_nodes.end());

	for (auto *remove_node : remove_nodes)
	{
		for (auto iter = m_impl->nodes.begin(); iter != m_impl->nodes.end(); iter++)
		{
			if (iter->get() == remove_node)
			{
				m_impl->nodes.erase(iter);
				break;
			}
		}
	}
}

std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> &Scene::GetComponents()
{
	return m_impl->components;
}

void Scene::Save(OutputArchive &archive)
{
	archive(m_impl->name);

	// Nodes
	{
		archive(m_impl->nodes.size());
		for (auto &node : m_impl->nodes)
		{
			node->Save(archive);
		}
		for (auto &node : m_impl->nodes)
		{
			archive(node->GetParent() ? node->GetParent()->GetID() : ~0ull);
		}
	}

	// Components
	{
		archive(m_impl->components.size());
		for (auto &[type, components] : m_impl->components)
		{
			archive(std::string(type.name()));
			archive(components.size());
			for (auto &component : components)
			{
				archive(component->GetNode()->GetID());
				component->Save(archive);
			}
		}
	}
}

void Scene::Load(InputArchive &archive)
{
	archive(m_impl->name);

	std::unordered_map<size_t, Node *> node_map;
	// Nodes
	{
		m_impl->nodes.clear();
		size_t node_count = 0;
		archive(node_count);
		for (size_t i = 0; i < node_count; i++)
		{
			std::unique_ptr<Node> node = std::make_unique<Node>(0, *this);
			node->Load(archive);
			m_impl->nodes.emplace_back(std::move(node));
			node_map[m_impl->nodes.back()->GetID()] = m_impl->nodes.back().get();
		}
		for (auto &node : m_impl->nodes)
		{
			size_t parent = ~0ull;
			archive(parent);
			if (parent != ~0ull)
			{
				node->SetParent(node_map[parent]);
			}
		}
	}

	// Components
	{
		const std::unordered_map<std::string, std::function<void(Node *&, InputArchive &)>> load_component_map = {
#define LOAD_COMPONENT(Cmpt)                                                                                                                   \
	{                                                                                                                                          \
		typeid(Cmpt).name(), [](Node *&node, InputArchive &archive) { node->AddComponent<Cmpt>(std::make_unique<Cmpt>(node))->Load(archive); } \
	}

		    LOAD_COMPONENT(Cmpt::Transform),
		    LOAD_COMPONENT(Cmpt::PerspectiveCamera),
		    LOAD_COMPONENT(Cmpt::OrthographicCamera),
		    LOAD_COMPONENT(Cmpt::MeshRenderer),
		    LOAD_COMPONENT(Cmpt::SkinnedMeshRenderer),
		    LOAD_COMPONENT(Cmpt::SpotLight),
		    LOAD_COMPONENT(Cmpt::PointLight),
		    LOAD_COMPONENT(Cmpt::DirectionalLight),
		    LOAD_COMPONENT(Cmpt::RectLight),
		    LOAD_COMPONENT(Cmpt::EnvironmentLight),
		};

		size_t component_type_count = 0;
		archive(component_type_count);
		for (size_t i = 0; i < component_type_count; i++)
		{
			std::string type_name = "";
			archive(type_name);
			size_t compoent_count = 0;
			archive(compoent_count);
			for (size_t j = 0; j < compoent_count; j++)
			{
				size_t node_id = 0;
				archive(node_id);
				Node *node = node_map[node_id];
				load_component_map.at(type_name)(node, archive);
			}
		}
	}
}

void Scene::Clear()
{
	auto roots = GetRoots();
	for (auto &root : roots)
	{
		EraseNode(root);
	}
}

bool Scene::IsUpdate() const
{
	return m_impl->update;
}

void Scene::Update(bool update)
{
	if (update)
	{
		m_impl->update = true;
	}
	else
	{
		for (auto &[type, cmpts] : m_impl->components)
		{
			for (auto &cmpt : cmpts)
			{
				m_impl->update |= cmpt->IsUpdate();
				cmpt->SetUpdate(false);
			}
		}
	}
}

void Scene::Reset()
{
	m_impl->update = false;
}
}        // namespace Ilum