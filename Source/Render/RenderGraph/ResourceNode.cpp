#pragma once

#include "ResourceNode.hpp"
#include "PassNode.hpp"
#include "RenderGraph.hpp"

namespace Ilum::Render
{
IResourceNode::IResourceNode(const std::string &name, RenderGraph &render_graph) :
    RenderNode(name, render_graph), m_write_id(NewUUID()), m_read_id(NewUUID())
{
	m_render_graph.RegisterPin(m_write_id, m_uuid);
	m_render_graph.RegisterPin(m_read_id, m_uuid);
}
}        // namespace Ilum::Render