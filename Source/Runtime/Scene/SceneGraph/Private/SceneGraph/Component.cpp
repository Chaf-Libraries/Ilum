#include "Component.hpp"
#include "Node.hpp"

namespace Ilum
{
Component::Component(const std::string &name, Node *node) :
    m_name(name), p_node(node)
{
}

const std::string &Component::GetName()
{
	return m_name;
}
}        // namespace Ilum