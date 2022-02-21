#pragma once

#include "RenderNode.hpp"

namespace Ilum::Render
{
static uint64_t GLOBAL_UUID = 0;
	
RenderNode::RenderNode() :
	m_uuid(GLOBAL_UUID++)
{

}

uint64_t RenderNode::GetUUID() const
{
	return m_uuid;
}

uint64_t RenderNode::NewUUID()
{
	return GLOBAL_UUID++;
}
}        // namespace Ilum::Render