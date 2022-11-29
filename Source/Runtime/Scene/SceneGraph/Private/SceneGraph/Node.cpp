#include "Node.hpp"
#include "Component.hpp"
#include "Scene.hpp"

namespace Ilum
{
struct Node::Impl
{
	Impl(Scene &scene) :
	    scene(scene)
	{
	}

	~Impl() = default;

	Scene &scene;

	size_t id = ~0U;

	std::string name;

	Node *parent = nullptr;

	std::vector<Node *> children;

	std::unordered_map<std::type_index, Component *> components;
};

Node::Node(size_t id, Scene &scene, const std::string &name)
{
	m_impl       = new Impl(scene);
	m_impl->id   = id;
	m_impl->name = name;
}

Node::~Node()
{
	for (auto &[type, cmpt] : m_impl->components)
	{
		auto &scene_components = m_impl->scene.GetComponents()[type];
		for (auto iter = scene_components.begin(); iter != scene_components.end(); iter++)
		{
			if (iter->get() == cmpt)
			{
				scene_components.erase(iter);
				break;
			}
		}
	}

	delete m_impl;
}

size_t Node::GetID() const
{
	return m_impl->id;
}

void Node::SetName(const std::string &name)
{
	m_impl->name = name;
}

const std::string &Node::GetName() const
{
	return m_impl->name;
}

Node *Node::GetParent()
{
	return m_impl->parent;
}

void Node::SetParent(Node *node)
{
	if (m_impl->parent)
	{
		m_impl->parent->EraseChild(this);
	}

	m_impl->parent = node;

	if (m_impl->parent)
	{
		m_impl->parent->AddChild(this);
	}
}

const std::vector<Node *> &Node::GetChildren() const
{
	return m_impl->children;
}

const std::unordered_map<std::type_index, Component *> &Node::GetComponents() const
{
	return m_impl->components;
}

void Node::EraseChild(Node *node)
{
	auto iter = std::find(m_impl->children.begin(), m_impl->children.end(), node);
	if (iter != m_impl->children.end())
	{
		m_impl->children.erase(iter);
	}
}

void Node::AddChild(Node *node)
{
	m_impl->children.push_back(node);
}

bool Node::HasComponent_(std::type_index index)
{
	return m_impl->components.find(index) != m_impl->components.end();
}

Component *Node::GetComponent_(std::type_index index)
{
	return HasComponent_(index) ? m_impl->components.at(index) : nullptr;
}

Component *Node::AddComponent_(std::unique_ptr<Component> &&component)
{
	auto &ptr = m_impl->scene.GetComponents()[component->GetType()].emplace_back(std::move(component));
	m_impl->components.emplace(ptr->GetType(), ptr.get());
	return ptr.get();
}

void Node::AddComponent_(Component *component)
{
	m_impl->components.emplace(component->GetType(), component);
}

void Node::EraseComponent(std::type_index index)
{
	auto cmpt_iter = m_impl->components.find(index);

	if (cmpt_iter == m_impl->components.end())
	{
		return;
	}

	auto &cmpt_list = m_impl->scene.GetComponents()[index];

	for (auto iter = cmpt_list.begin(); iter != cmpt_list.end(); iter++)
	{
		if (iter->get() == cmpt_iter->second)
		{
			cmpt_list.erase(iter);
			break;
		}
	}

	m_impl->components.erase(cmpt_iter);
}
}        // namespace Ilum