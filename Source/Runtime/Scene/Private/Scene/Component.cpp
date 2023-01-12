#include "Component.hpp"
#include "Node.hpp"

namespace Ilum
{
 //bool Component::m_update = false;

Component::Component(const char *name, Node *node) :
    m_name(name), p_node(node)
{
	m_update = true;
}

Component::~Component()
{
	m_update = true;
}

const char *Component::GetName() const
{
	return m_name;
}

Node *Component::GetNode() const
{
	return p_node;
}

bool Component::IsUpdate()
{
	return m_update;
}

void Component::Update(bool update)
{
	m_update = update;
}
}        // namespace Ilum