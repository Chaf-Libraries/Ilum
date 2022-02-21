#pragma once

#include "ResourceNode.hpp"
#include "PassNode.hpp"

namespace Ilum::Render
{
IResourceNode::IResourceNode(const std::string &name) :
    m_name(name), m_node_id(NewUUID()), m_write_id(NewUUID()), m_read_id(NewUUID())
{
}

const std::string &IResourceNode::GetName() const
{
	return m_name;
}
}        // namespace Ilum::Render