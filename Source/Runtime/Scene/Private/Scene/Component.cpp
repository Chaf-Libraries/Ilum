#include "Component.hpp"
#include "Node.hpp"

namespace Ilum
{
Component::Component(const char *name, Node *node) :
    m_name(name), p_node(node)
{
}

const char *Component::GetName() const
{
	return m_name;
}

Node *Component::GetNode() const
{
	return p_node;
}
}        // namespace Ilum