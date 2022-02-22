#pragma once

#include "RenderNode.hpp"

namespace Ilum::Render
{
static int32_t GLOBAL_UUID = 0;
	
RenderNode::RenderNode(const std::string &name, RenderGraph &render_graph) :
	m_uuid(GLOBAL_UUID++),
    m_name(name),
    m_render_graph(render_graph)
{

}

int32_t RenderNode::GetUUID() const
{
	return m_uuid;
}

const std::string &RenderNode::GetName() const
{
	return m_name;
}

int32_t RenderNode::NewUUID()
{
	return GLOBAL_UUID++;
}
}        // namespace Ilum::Render