#pragma once

#include "BindlessTextureNode.hpp"

#include <imnodes.h>

namespace Ilum::Render
{
BindlessTextureNode::BindlessTextureNode():
    IResourceNode("Bindless Texture Node")
{
}

void BindlessTextureNode::OnImGui()
{
}

void BindlessTextureNode::OnImNode()
{
	ImNodes::BeginNode(m_uuid);

	ImNodes::BeginNodeTitleBar();
	ImGui::TextUnformatted(m_name.c_str());
	ImNodes::EndNodeTitleBar();

	ImNodes::BeginOutputAttribute(m_read_id);
	ImGui::Indent(40);
	ImGui::Text("output");
	ImNodes::EndOutputAttribute();

	ImNodes::EndNode();
}

void BindlessTextureNode::OnUpdate()
{
}
}        // namespace Ilum::Render