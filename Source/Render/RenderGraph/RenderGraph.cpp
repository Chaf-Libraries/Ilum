#include "RenderGraph.hpp"

#include <imnodes.h>

namespace Ilum::Render
{
void RenderGraph::RegisterPin(int32_t uuid, PinDesc *pin_desc)
{
	m_pin_desc[uuid] = pin_desc;
}

void RenderGraph::UnRegisterPin(int32_t uuid)
{
	if (m_pin_desc.find(uuid) != m_pin_desc.end())
	{
		m_pin_desc.erase(uuid);
	}
	for (auto iter = m_links.begin(); iter != m_links.end();)
	{
		if (iter->first == uuid || iter->second == uuid)
		{
			iter = m_links.erase(iter);
		}
		else
		{
			iter++;
		}
	}
}

void RenderGraph::OnImGui()
{
	ImNodes::BeginNodeEditor();

	for (auto& [uuid, node] : m_render_nodes)
	{
		node->OnImNode();
	}

	// Link
	for (int i = 0; i < m_links.size(); ++i)
	{
		const std::pair<int, int> p = m_links[i];
		ImNodes::Link(i, p.first, p.second);
	}

	ImNodes::MiniMap();
	ImNodes::EndNodeEditor();

	int link;
	if (ImNodes::IsLinkHovered(&link))
	{
		if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
		{
			m_links.erase(m_links.begin() + link);
		}
	}

	int start_attr, end_attr;
	if (ImNodes::IsLinkCreated(&start_attr, &end_attr))
	{
		if (m_pin_desc[start_attr]->Hash() == m_pin_desc[end_attr]->Hash())
		{
			m_links.push_back(std::make_pair(start_attr, end_attr));
		}
	}
}
}        // namespace Ilum::Render