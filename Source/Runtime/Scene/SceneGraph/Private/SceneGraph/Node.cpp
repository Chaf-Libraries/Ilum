#include "Node.hpp"
#include "Component.hpp"
#include "Scene.hpp"

namespace Ilum
{
Node::Node(size_t id, Scene &scene, const std::string &name) :
    m_id(id), m_scene(scene), m_name(name)
{
}

Node::~Node()
{
	for (auto &[type, cmpt] : m_components)
	{
		auto &scene_components = m_scene.m_components[type];
		for (auto iter = scene_components.begin(); iter != scene_components.end(); iter++)
		{
			if (iter->get() == cmpt)
			{
				scene_components.erase(iter);
				break;
			}
		}
	}
}

size_t Node::GetID() const
{
	return m_id;
}

void Node::SetName(const std::string &name)
{
	m_name = name;
}

const std::string &Node::GetName() const
{
	return m_name;
}

Node *Node::GetParent()
{
	return m_parent;
}

void Node::SetParent(Node *node)
{
	if (m_parent)
	{
		m_parent->EraseChild(this);
	}

	m_parent = node;

	if (m_parent)
	{
		m_parent->AddChild(this);
	}
}

const std::vector<Node *> &Node::GetChildren() const
{
	return m_children;
}

const std::unordered_map<std::type_index, Component *> &Node::GetComponents() const
{
	return m_components;
}

void Node::EraseChild(Node *node)
{
	auto iter = std::find(m_children.begin(), m_children.end(), node);
	if (iter != m_children.end())
	{
		m_children.erase(iter);
	}
}

void Node::AddChild(Node *node)
{
	m_children.push_back(node);
}

bool Node::HasComponent_(std::type_index index)
{
	return m_components.find(index) != m_components.end();
}

Component *Node::GetComponent_(std::type_index index)
{
	return HasComponent_(index) ? m_components.at(index) : nullptr;
}

Component *Node::AddComponent_(std::unique_ptr<Component> &&component)
{
	auto &ptr = m_scene.m_components[component->GetType()].emplace_back(std::move(component));
	m_components.emplace(ptr->GetType(), ptr.get());
	return ptr.get();
}

void Node::AddComponent_(Component *component)
{
	m_components.emplace(component->GetType(), component);
}

void Node::EraseComponent(std::type_index index)
{
	auto cmpt_iter = m_components.find(index);

	if (cmpt_iter == m_components.end())
	{
		return;
	}

	auto &cmpt_list = m_scene.m_components[index];

	for (auto iter = cmpt_list.begin(); iter != cmpt_list.end(); iter++)
	{
		if (iter->get() == cmpt_iter->second)
		{
			cmpt_list.erase(iter);
			break;
		}
	}

	m_components.erase(cmpt_iter);
}
}        // namespace Ilum